#!/usr/bin/env python3
"""Convert split ROS 2 MCAP camera/IMU bags to OKVIS' EuRoC layout."""

import argparse
import csv
import sys
from itertools import chain
from pathlib import Path


CAMERA_CSV_HEADER = ("#timestamp [ns]", "filename")
IMU_CSV_HEADER = (
    "#timestamp [ns]",
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert cam0...camN and imu split MCAP bags to the EuRoC-style "
            "directory consumed by okvis_app_synchronous."
        )
    )
    parser.add_argument("input", type=Path, help="Sequence directory containing cam*/ and imu/")
    parser.add_argument("output", type=Path, help="New EuRoC-style output directory")
    parser.add_argument(
        "--cameras", type=int, default=4, help="Number of cameras (default: 4)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(960, 600),
        help="Output image size matching the calibration (default: 960 600)",
    )
    parser.add_argument(
        "--camera-topic-template",
        default="/cam{index}/image/compressed",
        help="Camera topic template; {index} is replaced by the camera index",
    )
    parser.add_argument("--imu-topic", default="/imu/data_raw", help="IMU topic name")
    parser.add_argument(
        "--sync-tolerance-ms",
        type=float,
        default=10.0,
        help="Maximum accepted inter-camera timestamp skew (default: 10 ms)",
    )
    parser.add_argument(
        "--png-compression",
        type=int,
        choices=range(10),
        default=1,
        metavar="0..9",
        help="OpenCV PNG compression level (default: 1)",
    )
    args = parser.parse_args()
    if args.cameras <= 0:
        parser.error("--cameras must be positive")
    if any(value <= 0 for value in args.image_size):
        parser.error("--image-size values must be positive")
    if args.sync_tolerance_ms < 0:
        parser.error("--sync-tolerance-ms must be non-negative")
    return args


def load_dependencies():
    try:
        import cv2
        import numpy as np
        import rosbag2_py
        from rclpy.serialization import deserialize_message
        from sensor_msgs.msg import CompressedImage, Imu
    except ImportError as error:
        raise RuntimeError(
            "ROS 2 Python, sensor_msgs, OpenCV, and NumPy are required. "
            "Source /opt/ros/humble/setup.bash before running this tool."
        ) from error
    return cv2, np, rosbag2_py, deserialize_message, CompressedImage, Imu


def message_timestamp_ns(message, recorded_timestamp):
    timestamp = message.header.stamp.sec * 1_000_000_000 + message.header.stamp.nanosec
    return timestamp if timestamp > 0 else recorded_timestamp


def open_reader(rosbag2_py, bag_path):
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=str(bag_path), storage_id="mcap")
    converter_options = rosbag2_py.ConverterOptions("", "")
    try:
        reader.open(storage_options, converter_options)
    except RuntimeError as error:
        raise RuntimeError(
            f"Cannot open MCAP bag {bag_path}. Install "
            "ros-humble-rosbag2-storage-mcap and source the ROS 2 environment."
        ) from error
    return reader


def deserialize_topic_messages(reader, topic, message_type, deserialize_message):
    while reader.has_next():
        message_topic, serialized, recorded_timestamp = reader.read_next()
        if message_topic == topic:
            yield deserialize_message(serialized, message_type), recorded_timestamp


def validate_inputs(input_path, output_path, num_cameras):
    if not input_path.is_dir():
        raise RuntimeError(f"Input directory does not exist: {input_path}")
    required = [input_path / f"cam{index}" for index in range(num_cameras)]
    required.append(input_path / "imu")
    missing = [path for path in required if not (path / "metadata.yaml").is_file()]
    if missing:
        raise RuntimeError("Missing bag metadata: " + ", ".join(str(path) for path in missing))
    if output_path.exists():
        raise RuntimeError(f"Output already exists; refusing to overwrite: {output_path}")
    if output_path.resolve().is_relative_to(input_path.resolve()):
        raise RuntimeError("Output directory must not be inside the input sequence")


def convert_camera(
    bag_path,
    output_path,
    topic,
    image_size,
    png_compression,
    dependencies,
):
    cv2, np, rosbag2_py, deserialize_message, CompressedImage, _ = dependencies
    reader = open_reader(rosbag2_py, bag_path)
    data_path = output_path / "data"
    data_path.mkdir(parents=True)
    messages = deserialize_topic_messages(
        reader,
        topic,
        CompressedImage,
        deserialize_message,
    )
    first = next(messages, None)
    if first is None:
        raise RuntimeError(f"Topic {topic} not found in {bag_path}")

    image_format = first[0].format.lower().strip()
    if "h264" in image_format:
        return convert_h264_camera(
            chain((first,), messages),
            bag_path,
            output_path,
            data_path,
            image_size,
            png_compression,
            cv2,
        )
    return convert_still_image_camera(
        chain((first,), messages),
        bag_path,
        output_path,
        data_path,
        image_size,
        png_compression,
        cv2,
        np,
    )


def camera_frame_info(message, recorded_timestamp, previous_timestamp, duplicate_count):
    timestamp = message_timestamp_ns(message, recorded_timestamp)
    if timestamp < previous_timestamp:
        raise RuntimeError("Non-monotonic camera timestamps")
    if timestamp == previous_timestamp:
        duplicate_count += 1
        filename = f"{timestamp}_{duplicate_count}.png"
    else:
        duplicate_count = 0
        filename = f"{timestamp}.png"
    return timestamp, filename, duplicate_count


def write_camera_image(cv2, image, path, image_size, png_compression):
    if (image.shape[1], image.shape[0]) != tuple(image_size):
        image = cv2.resize(image, tuple(image_size), interpolation=cv2.INTER_AREA)
    written = cv2.imwrite(
        str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    )
    if not written:
        raise RuntimeError(f"Failed to write {path}")


def convert_still_image_camera(
    messages,
    bag_path,
    output_path,
    data_path,
    image_size,
    png_compression,
    cv2,
    np,
):
    timestamps = []
    previous_timestamp = -1
    duplicate_count = 0
    original_size = None

    with (output_path / "data.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(CAMERA_CSV_HEADER)
        for message, recorded_timestamp in messages:
            timestamp, filename, duplicate_count = camera_frame_info(
                message, recorded_timestamp, previous_timestamp, duplicate_count
            )
            encoded = np.frombuffer(message.data, dtype=np.uint8)
            image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise RuntimeError(f"Failed to decode image at {timestamp} from {bag_path}")
            if original_size is None:
                original_size = (image.shape[1], image.shape[0])
            write_camera_image(
                cv2, image, data_path / filename, image_size, png_compression
            )
            writer.writerow((timestamp, filename))
            timestamps.append(timestamp)
            previous_timestamp = timestamp
            if len(timestamps) % 500 == 0:
                print(f"  {bag_path.name}: {len(timestamps)} images", flush=True)

    print(
        f"  {bag_path.name}: wrote {len(timestamps)} images, "
        f"{original_size[0]}x{original_size[1]} -> {image_size[0]}x{image_size[1]}"
    )
    return timestamps


def convert_h264_camera(
    messages,
    bag_path,
    output_path,
    data_path,
    image_size,
    png_compression,
    cv2,
):
    stream_path = output_path / "stream.h264"
    frame_info = []
    previous_timestamp = -1
    duplicate_count = 0

    print(f"  {bag_path.name}: collecting H.264 stream...", flush=True)
    with stream_path.open("wb") as stream:
        for message, recorded_timestamp in messages:
            timestamp, filename, duplicate_count = camera_frame_info(
                message, recorded_timestamp, previous_timestamp, duplicate_count
            )
            stream.write(bytes(message.data))
            frame_info.append((timestamp, filename))
            previous_timestamp = timestamp
            if len(frame_info) % 500 == 0:
                print(f"  {bag_path.name}: collected {len(frame_info)} frames", flush=True)

    capture = cv2.VideoCapture(str(stream_path), cv2.CAP_FFMPEG)
    if not capture.isOpened():
        raise RuntimeError(
            f"OpenCV's FFmpeg backend cannot open the H.264 stream from {bag_path}"
        )

    original_size = None
    try:
        with (output_path / "data.csv").open("w", newline="") as csv_file:
            writer = csv.writer(csv_file, lineterminator="\n")
            writer.writerow(CAMERA_CSV_HEADER)
            for index, (timestamp, filename) in enumerate(frame_info, start=1):
                decoded, image = capture.read()
                if not decoded:
                    raise RuntimeError(
                        f"H.264 stream ended after {index - 1} of {len(frame_info)} frames "
                        f"in {bag_path}"
                    )
                if original_size is None:
                    original_size = (image.shape[1], image.shape[0])
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                write_camera_image(
                    cv2, image, data_path / filename, image_size, png_compression
                )
                writer.writerow((timestamp, filename))
                if index % 500 == 0:
                    print(f"  {bag_path.name}: wrote {index} images", flush=True)
            extra_frame, _ = capture.read()
            if extra_frame:
                raise RuntimeError(
                    f"H.264 stream contains more decoded frames than messages in {bag_path}"
                )
    finally:
        capture.release()

    stream_path.unlink()
    print(
        f"  {bag_path.name}: wrote {len(frame_info)} H.264 images, "
        f"{original_size[0]}x{original_size[1]} -> {image_size[0]}x{image_size[1]}"
    )
    return [timestamp for timestamp, _ in frame_info]


def convert_imu(bag_path, output_path, topic, dependencies):
    _, _, rosbag2_py, deserialize_message, _, Imu = dependencies
    reader = open_reader(rosbag2_py, bag_path)
    output_path.mkdir(parents=True)
    count = 0
    previous_timestamp = -1

    with (output_path / "data.csv").open("w", newline="") as csv_file:
        writer = csv.writer(csv_file, lineterminator="\n")
        writer.writerow(IMU_CSV_HEADER)
        while reader.has_next():
            message_topic, serialized, recorded_timestamp = reader.read_next()
            if message_topic != topic:
                continue
            message = deserialize_message(serialized, Imu)
            timestamp = message_timestamp_ns(message, recorded_timestamp)
            if timestamp < previous_timestamp:
                raise RuntimeError(f"Non-monotonic timestamps in {bag_path}")
            writer.writerow(
                (
                    timestamp,
                    format(message.angular_velocity.x, ".17g"),
                    format(message.angular_velocity.y, ".17g"),
                    format(message.angular_velocity.z, ".17g"),
                    format(message.linear_acceleration.x, ".17g"),
                    format(message.linear_acceleration.y, ".17g"),
                    format(message.linear_acceleration.z, ".17g"),
                )
            )
            previous_timestamp = timestamp
            count += 1

    if count == 0:
        raise RuntimeError(f"Topic {topic} not found in {bag_path}")
    print(f"  {bag_path.name}: wrote {count} IMU measurements")
    return count


def validate_camera_sync(camera_timestamps, tolerance_ms):
    counts = [len(timestamps) for timestamps in camera_timestamps]
    if not counts or any(count == 0 for count in counts):
        raise RuntimeError(f"Cannot synchronize empty camera streams: {counts}")

    tolerance_ns = round(tolerance_ms * 1_000_000)
    positions = [0] * len(camera_timestamps)
    dropped_frames = [0] * len(camera_timestamps)
    matched_groups = 0
    max_matched_skew_ns = 0

    while all(position < count for position, count in zip(positions, counts)):
        current = [
            camera_timestamps[index][position]
            for index, position in enumerate(positions)
        ]
        skew_ns = max(current) - min(current)
        if skew_ns <= tolerance_ns:
            matched_groups += 1
            max_matched_skew_ns = max(max_matched_skew_ns, skew_ns)
            positions = [position + 1 for position in positions]
        else:
            oldest_camera = min(range(len(current)), key=current.__getitem__)
            positions[oldest_camera] += 1
            dropped_frames[oldest_camera] += 1

    for index, (position, count) in enumerate(zip(positions, counts)):
        dropped_frames[index] += count - position

    if matched_groups == 0:
        raise RuntimeError(
            f"No camera frames can be synchronized within {tolerance_ms:.3f} ms"
        )
    print(
        f"Camera synchronization: {matched_groups} matched groups, "
        f"dropped frames {dropped_frames}, maximum matched skew "
        f"{max_matched_skew_ns / 1e6:.3f} ms"
    )
    return {
        "matched_groups": matched_groups,
        "dropped_frames": dropped_frames,
        "max_matched_skew_ns": max_matched_skew_ns,
    }


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    temporary_path = output_path.with_name(output_path.name + ".incomplete")

    validate_inputs(input_path, output_path, args.cameras)
    if temporary_path.exists():
        raise RuntimeError(
            f"Incomplete output already exists: {temporary_path}. Remove it before retrying."
        )
    dependencies = load_dependencies()

    temporary_path.mkdir(parents=True)
    try:
        camera_timestamps = []
        for index in range(args.cameras):
            print(f"Converting camera {index}...")
            topic = args.camera_topic_template.format(index=index)
            timestamps = convert_camera(
                input_path / f"cam{index}",
                temporary_path / f"cam{index}",
                topic,
                tuple(args.image_size),
                args.png_compression,
                dependencies,
            )
            camera_timestamps.append(timestamps)

        print("Converting IMU...")
        convert_imu(input_path / "imu", temporary_path / "imu0", args.imu_topic, dependencies)
        validate_camera_sync(camera_timestamps, args.sync_tolerance_ms)
        (temporary_path / ".complete").touch()
        temporary_path.rename(output_path)
    except Exception:
        print(f"Conversion failed; partial output kept at {temporary_path}", file=sys.stderr)
        raise

    print(f"Conversion complete: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
