# Repository Guidelines

## Project Structure & Module Organization

OKVIS2-X is a C++17 CMake project organized as small libraries. Core modules live in
`okvis_*` directories; public headers are under `include/okvis`, implementations under
`src`, and module tests under `test`. Executables and scripts are in `okvis_apps/`, while
ROS 2 nodes and launch files are in `okvis_ros2/`. Dataset and sensor settings belong in
`config/`; evaluation utilities are in `eval/`, and runtime models or sample media are in
`resources/`. `supereight2/` and `external/` contain bundled dependencies: avoid changing
them unless the change is intentionally dependency-specific.

## Build, Test, and Development Commands

Initialize dependencies after cloning:

```bash
git submodule update --init --recursive
```

For a dependency-light local build without ROS 2, Torch, or RealSense:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_ROS2=OFF \
  -DUSE_NN=OFF -DHAVE_LIBREALSENSE=OFF -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure
```

Release mode is strongly recommended for usable SLAM performance. From the workspace
root, `colcon build --cmake-args -DBUILD_ROS2=ON` builds the ROS 2 package. Use
`-DUSE_COLIDMAP=OFF` for LiDAR nodes; FindAnything requires both `USE_NN` and
`USE_COLIDMAP`.

## Coding Style & Naming Conventions

Follow the ETHZ-ASL C++ coding guidelines linked from `README.md` and preserve the style
of the module being edited. Use spaces, C++17 features, `CamelCase` types,
`lowerCamelCase` functions and variables, and trailing underscores for private data
members. Public headers use `.hpp`; implementations use `.cpp`. Keep includes grouped
and compile cleanly under the repository's `-Wall -Wpedantic -Wextra` flags. The
`supereight2/.clang-format` file applies to that subtree only.

## Testing Guidelines

Tests use GoogleTest and are enabled with `-DBUILD_TESTS=ON`. Add focused cases beside
the owning module, typically in `test/TestFeature.cpp`, and register new sources or
targets in that module's `CMakeLists.txt`. Name suites after the class or subsystem and
make case names describe behavior. Run the full CTest suite before submitting.

## Commit & Pull Request Guidelines

Recent commits use short, sentence-case summaries such as `Fix subsampling error` and
keep each commit focused. Pull requests are required for all changes and receive admin
review. Describe the behavior and configuration affected, link relevant issues, list
tests run, and include logs or screenshots for ROS/visualization changes. For bug fixes,
provide the configuration and dataset details needed to reproduce the problem.
