#include <okvis/FrontendDiagnostics.hpp>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace okvis {
namespace diagnostics {
namespace {

std::optional<double> quantile(const std::vector<double>& sortedValues,
                               const double fraction) {
  if (sortedValues.empty()) {
    return std::nullopt;
  }
  const double position =
      static_cast<double>(sortedValues.size() - 1) * fraction;
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double upperWeight = position - static_cast<double>(lower);
  return sortedValues[lower] * (1.0 - upperWeight) +
         sortedValues[upper] * upperWeight;
}

}  // namespace

void DiagnosticDistribution::add(const double value) {
  if (std::isfinite(value)) {
    values_.push_back(value);
  }
}

void DiagnosticDistribution::merge(const DiagnosticDistribution& other) {
  values_.insert(values_.end(), other.values_.begin(), other.values_.end());
}

DistributionSummary DiagnosticDistribution::summary() const {
  std::vector<double> sortedValues = values_;
  std::sort(sortedValues.begin(), sortedValues.end());
  DistributionSummary result;
  result.p10 = quantile(sortedValues, 0.1);
  result.median = quantile(sortedValues, 0.5);
  result.p90 = quantile(sortedValues, 0.9);
  return result;
}

void CameraMapMatchAccumulator::merge(
    const CameraMapMatchAccumulator& other) {
  projectedEligible += other.projectedEligible;
  descriptorComparisons += other.descriptorComparisons;
  descriptorCandidatesBelowThreshold +=
      other.descriptorCandidatesBelowThreshold;
  epipolarRejected += other.epipolarRejected;
  divergentRayRejected += other.divergentRayRejected;
  acceptedInitialised += other.acceptedInitialised;
  acceptedUninitialised += other.acceptedUninitialised;
  bestDescriptorDistance.merge(other.bestDescriptorDistance);
  acceptedDescriptorDistance.merge(other.acceptedDescriptorDistance);
  predictedReprojectionErrorPx.merge(other.predictedReprojectionErrorPx);
}

TriangulationAccumulator::TriangulationAccumulator(
    const TriangulationSource source, const int camera0, const int camera1) {
  counts_.source = source;
  counts_.camera0 = camera0;
  counts_.camera1 = camera1;
}

void TriangulationAccumulator::recordAttempt(
    const double baselineM, const double rayAngleRad,
    const double pixelDisplacementPx, const double depthM, const bool valid,
    const bool parallel) {
  ++counts_.attempts;
  if (valid) {
    ++counts_.valid;
  } else {
    ++counts_.invalid;
  }
  if (parallel) {
    ++counts_.parallel;
  }
  baselineM_.add(baselineM);
  rayAngleRad_.add(rayAngleRad);
  pixelDisplacementPx_.add(pixelDisplacementPx);
  depthM_.add(depthM);
}

void TriangulationAccumulator::recordDescriptorCandidate() {
  ++counts_.descriptorCandidates;
}

void TriangulationAccumulator::recordInitialisable() {
  ++counts_.initialisable;
}

void TriangulationAccumulator::recordLandmarkBirth() {
  ++counts_.landmarkBirths;
}

void TriangulationAccumulator::recordLandmarkInitialisation() {
  ++counts_.landmarkInitialisations;
}

void TriangulationAccumulator::recordBackProjectionRejected() {
  ++counts_.backProjectionRejected;
}

void TriangulationAccumulator::recordDescriptorRejected() {
  ++counts_.descriptorRejected;
}

void TriangulationAccumulator::recordEpipolarRejected() {
  ++counts_.epipolarRejected;
}

void TriangulationAccumulator::recordDivergentRaysRejected() {
  ++counts_.divergentRaysRejected;
}

void TriangulationAccumulator::recordDepthRejected() {
  ++counts_.depthRejected;
}

void TriangulationAccumulator::recordProjectionRejected() {
  ++counts_.projectionRejected;
}

void TriangulationAccumulator::recordReprojectionRejected() {
  ++counts_.reprojectionRejected;
}

void TriangulationAccumulator::merge(const TriangulationAccumulator& other) {
  if (counts_.source != other.counts_.source ||
      counts_.camera0 != other.counts_.camera0 ||
      counts_.camera1 != other.counts_.camera1) {
    return;
  }
  counts_.attempts += other.counts_.attempts;
  counts_.descriptorCandidates += other.counts_.descriptorCandidates;
  counts_.valid += other.counts_.valid;
  counts_.invalid += other.counts_.invalid;
  counts_.parallel += other.counts_.parallel;
  counts_.initialisable += other.counts_.initialisable;
  counts_.backProjectionRejected += other.counts_.backProjectionRejected;
  counts_.descriptorRejected += other.counts_.descriptorRejected;
  counts_.epipolarRejected += other.counts_.epipolarRejected;
  counts_.divergentRaysRejected += other.counts_.divergentRaysRejected;
  counts_.depthRejected += other.counts_.depthRejected;
  counts_.projectionRejected += other.counts_.projectionRejected;
  counts_.reprojectionRejected += other.counts_.reprojectionRejected;
  counts_.landmarkBirths += other.counts_.landmarkBirths;
  counts_.landmarkInitialisations += other.counts_.landmarkInitialisations;
  baselineM_.merge(other.baselineM_);
  rayAngleRad_.merge(other.rayAngleRad_);
  pixelDisplacementPx_.merge(other.pixelDisplacementPx_);
  depthM_.merge(other.depthM_);
}

size_t TriangulationAccumulator::attempts() const {
  return counts_.attempts;
}

size_t TriangulationAccumulator::parallel() const {
  return counts_.parallel;
}

TriangulationDiagnosticRecord TriangulationAccumulator::toRecord(
    const uint64_t timestampNs, const uint64_t frameId) const {
  assert(counts_.attempts >= counts_.valid + counts_.invalid);
  assert(counts_.parallel <= counts_.attempts);
  assert(counts_.initialisable <= counts_.valid);
  assert(counts_.landmarkInitialisations <= counts_.initialisable);
  TriangulationDiagnosticRecord record = counts_;
  record.timestampNs = timestampNs;
  record.frameId = frameId;
  record.baselineM = baselineM_.summary();
  record.rayAngleRad = rayAngleRad_.summary();
  record.pixelDisplacementPx = pixelDisplacementPx_.summary();
  record.depthM = depthM_.summary();
  return record;
}

FrontendDiagnosticFrames::FrontendDiagnosticFrames(const size_t cameraCount)
    : cameraCount_(cameraCount) {}

void FrontendDiagnosticFrames::updateDetection(
    const uint64_t timestampNs, const size_t cameraIndex,
    const CameraDetectionAccumulator& detection) {
  if (cameraIndex >= cameraCount_) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  FrontendFrameAccumulator& pending = pendingByTimestamp_[timestampNs];
  pending.timestampNs = timestampNs;
  pending.cameras.resize(cameraCount_);
  pending.mapMatching.resize(cameraCount_);
  pending.cameras[cameraIndex] = detection;
}

std::shared_ptr<FrontendFrameAccumulator> FrontendDiagnosticFrames::bindFrame(
    const uint64_t frameId, const uint64_t timestampNs) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto existing = byFrameId_.find(frameId);
  if (existing != byFrameId_.end()) {
    return existing->second;
  }

  FrontendFrameAccumulator accumulator;
  const auto pending = pendingByTimestamp_.find(timestampNs);
  if (pending != pendingByTimestamp_.end()) {
    accumulator = std::move(pending->second);
    pendingByTimestamp_.erase(pending);
  }
  accumulator.timestampNs = timestampNs;
  accumulator.frameId = frameId;
  accumulator.cameras.resize(cameraCount_);
  accumulator.mapMatching.resize(cameraCount_);
  accumulator.record.timestampNs = timestampNs;
  accumulator.record.frameId = frameId;
  auto shared =
      std::make_shared<FrontendFrameAccumulator>(std::move(accumulator));
  byFrameId_.emplace(frameId, shared);
  return shared;
}

std::optional<FrontendFrameAccumulator> FrontendDiagnosticFrames::take(
    const uint64_t frameId) {
  std::lock_guard<std::mutex> lock(mutex_);
  const auto frame = byFrameId_.find(frameId);
  if (frame == byFrameId_.end()) {
    return std::nullopt;
  }
  FrontendFrameAccumulator accumulator = std::move(*frame->second);
  byFrameId_.erase(frame);
  return accumulator;
}

void FrontendDiagnosticFrames::clear() {
  std::lock_guard<std::mutex> lock(mutex_);
  pendingByTimestamp_.clear();
  byFrameId_.clear();
}

CameraDetectionAccumulator summarizeKeypoints(
    const std::vector<cv::KeyPoint>& keypoints, const int imageWidth,
    const int imageHeight, const int gridColumns, const int gridRows) {
  CameraDetectionAccumulator result;
  result.keypoints = keypoints.size();
  for (const cv::KeyPoint& keypoint : keypoints) {
    result.response.add(static_cast<double>(keypoint.response));
  }
  if (imageWidth <= 0 || imageHeight <= 0 || gridColumns <= 0 ||
      gridRows <= 0 || keypoints.empty()) {
    return result;
  }

  std::vector<bool> occupied(
      static_cast<size_t>(gridColumns * gridRows), false);
  std::vector<cv::Point2f> points;
  points.reserve(keypoints.size());
  for (const cv::KeyPoint& keypoint : keypoints) {
    const int column = std::max(
        0, std::min(gridColumns - 1,
                    static_cast<int>(std::floor(
                        static_cast<double>(keypoint.pt.x) * gridColumns /
                        imageWidth))));
    const int row = std::max(
        0, std::min(gridRows - 1,
                    static_cast<int>(std::floor(
                        static_cast<double>(keypoint.pt.y) * gridRows /
                        imageHeight))));
    occupied[static_cast<size_t>(row * gridColumns + column)] = true;
    points.push_back(keypoint.pt);
  }
  const size_t occupiedCells = static_cast<size_t>(
      std::count(occupied.begin(), occupied.end(), true));
  result.occupiedGridFraction =
      static_cast<double>(occupiedCells) / occupied.size();

  if (points.size() >= 3) {
    std::vector<cv::Point2f> hull;
    cv::convexHull(points, hull);
    const double imageArea =
        static_cast<double>(imageWidth) * static_cast<double>(imageHeight);
    result.convexHullFraction = std::max(
        0.0, std::min(1.0, cv::contourArea(hull) / imageArea));
  }
  return result;
}

double cameraBaseline(const Eigen::Vector3d& center0,
                      const Eigen::Vector3d& center1) {
  return (center1 - center0).norm();
}

std::optional<double> rayAngle(const Eigen::Vector3d& ray0,
                               const Eigen::Vector3d& ray1) {
  const double denominator = ray0.norm() * ray1.norm();
  if (!std::isfinite(denominator) || denominator <= 0.0) {
    return std::nullopt;
  }
  const double cosine =
      std::max(-1.0, std::min(1.0, ray0.dot(ray1) / denominator));
  return std::acos(cosine);
}

double pixelDisplacement(const Eigen::Vector2d& point0,
                         const Eigen::Vector2d& point1) {
  return (point1 - point0).norm();
}

InitialisationModelOutcome classifyInitialisationModels(
    const size_t correspondences, const bool rotationModelComputed,
    const size_t rotationInliers, const bool relativePoseModelComputed,
    const size_t relativePoseInliers) {
  InitialisationModelOutcome outcome;
  if (correspondences < 10) {
    outcome.selection =
        InitialisationModelSelection::InsufficientCorrespondences;
    return outcome;
  }
  if (!rotationModelComputed && !relativePoseModelComputed) {
    return outcome;
  }

  const double denominator = static_cast<double>(correspondences);
  const double rotationRatio =
      rotationModelComputed ? static_cast<double>(rotationInliers) / denominator
                            : 0.0;
  const double relativePoseRatio =
      relativePoseModelComputed
          ? static_cast<double>(relativePoseInliers) / denominator
          : 0.0;
  if (rotationModelComputed &&
      (!relativePoseModelComputed || rotationRatio > relativePoseRatio ||
       rotationRatio > 0.8)) {
    outcome.selection = InitialisationModelSelection::RotationOnly;
    outcome.selectedInliers = rotationInliers;
    outcome.successful = rotationInliers > 10;
    return outcome;
  }
  if (relativePoseModelComputed) {
    outcome.selection = InitialisationModelSelection::RelativePose;
    outcome.selectedInliers = relativePoseInliers;
    outcome.successful = relativePoseInliers > 10 && relativePoseRatio > 0.8;
  }
  return outcome;
}

RansacOutcome classifyRansacOutcome(const bool hasPriorFrame,
                                    const size_t correspondences,
                                    const size_t inliers,
                                    const bool modelComputed) {
  RansacOutcome outcome;
  if (!hasPriorFrame) {
    return outcome;
  }
  if (correspondences < 10) {
    outcome.status = RansacStatus::InsufficientCorrespondences;
    outcome.returnedSuccess = static_cast<bool>(correspondences);
    return outcome;
  }
  if (!modelComputed) {
    outcome.status = RansacStatus::ModelComputationFailed;
    return outcome;
  }
  const double ratio =
      static_cast<double>(inliers) / static_cast<double>(correspondences);
  outcome.thresholdSuccess = inliers >= 10 && ratio > 0.7;
  outcome.returnedSuccess = outcome.thresholdSuccess;
  outcome.status = outcome.thresholdSuccess ? RansacStatus::ThresholdAccepted
                                            : RansacStatus::ThresholdRejected;
  return outcome;
}

}  // namespace diagnostics
}  // namespace okvis
