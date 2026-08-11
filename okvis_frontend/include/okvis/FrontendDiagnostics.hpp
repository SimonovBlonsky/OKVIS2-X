#ifndef OKVIS_FRONTENDDIAGNOSTICS_HPP
#define OKVIS_FRONTENDDIAGNOSTICS_HPP

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core/types.hpp>

#include <okvis/VioDiagnostics.hpp>

namespace okvis {
namespace diagnostics {

class DiagnosticDistribution {
 public:
  void add(double value);
  void merge(const DiagnosticDistribution& other);
  DistributionSummary summary() const;

 private:
  std::vector<double> values_;
};

struct CameraDetectionAccumulator {
  size_t keypoints = 0;
  DiagnosticDistribution response;
  double occupiedGridFraction = 0.0;
  double convexHullFraction = 0.0;
};

struct CameraMapMatchAccumulator {
  size_t projectedEligible = 0;
  size_t descriptorComparisons = 0;
  size_t descriptorCandidatesBelowThreshold = 0;
  size_t epipolarRejected = 0;
  size_t divergentRayRejected = 0;
  size_t acceptedInitialised = 0;
  size_t acceptedUninitialised = 0;
  DiagnosticDistribution bestDescriptorDistance;
  DiagnosticDistribution acceptedDescriptorDistance;
  DiagnosticDistribution predictedReprojectionErrorPx;

  void merge(const CameraMapMatchAccumulator& other);
};

class TriangulationAccumulator {
 public:
  TriangulationAccumulator(TriangulationSource source, int camera0,
                           int camera1);

  void recordAttempt(double baselineM, double rayAngleRad,
                     double pixelDisplacementPx, double depthM, bool valid,
                     bool parallel);
  void recordDescriptorCandidate();
  void recordInitialisable();
  void recordLandmarkBirth();
  void recordLandmarkInitialisation();
  void recordBackProjectionRejected();
  void recordDescriptorRejected();
  void recordEpipolarRejected();
  void recordDivergentRaysRejected();
  void recordDepthRejected();
  void recordProjectionRejected();
  void recordReprojectionRejected();
  void merge(const TriangulationAccumulator& other);
  size_t attempts() const;
  size_t parallel() const;
  TriangulationDiagnosticRecord toRecord(uint64_t timestampNs,
                                         uint64_t frameId) const;

 private:
  TriangulationDiagnosticRecord counts_;
  DiagnosticDistribution baselineM_;
  DiagnosticDistribution rayAngleRad_;
  DiagnosticDistribution pixelDisplacementPx_;
  DiagnosticDistribution depthM_;
};

struct FrontendFrameAccumulator {
  uint64_t timestampNs = 0;
  uint64_t frameId = 0;
  std::vector<CameraDetectionAccumulator> cameras;
  std::vector<CameraMapMatchAccumulator> mapMatching;
  DiagnosticDistribution descriptorDistance;
  DiagnosticDistribution predictedReprojectionErrorPx;
  PoseSnapshot dataAssociationStartPose;
  FrameDiagnosticRecord record;
  std::map<std::tuple<TriangulationSource, int, int>,
           TriangulationAccumulator>
      triangulation;
  std::vector<InitialisationDiagnosticRecord> initialisation;
  std::vector<RansacDiagnosticRecord> ransac;
};

class FrontendDiagnosticFrames {
 public:
  explicit FrontendDiagnosticFrames(size_t cameraCount);

  void updateDetection(uint64_t timestampNs, size_t cameraIndex,
                       const CameraDetectionAccumulator& detection);
  std::shared_ptr<FrontendFrameAccumulator> bindFrame(uint64_t frameId,
                                                      uint64_t timestampNs);
  std::optional<FrontendFrameAccumulator> take(uint64_t frameId);
  void clear();

 private:
  size_t cameraCount_;
  std::mutex mutex_;
  std::map<uint64_t, FrontendFrameAccumulator> pendingByTimestamp_;
  std::map<uint64_t, std::shared_ptr<FrontendFrameAccumulator>> byFrameId_;
};

CameraDetectionAccumulator summarizeKeypoints(
    const std::vector<cv::KeyPoint>& keypoints, int imageWidth, int imageHeight,
    int gridColumns = 4, int gridRows = 4);

double cameraBaseline(const Eigen::Vector3d& center0,
                      const Eigen::Vector3d& center1);
std::optional<double> rayAngle(const Eigen::Vector3d& ray0,
                               const Eigen::Vector3d& ray1);
double pixelDisplacement(const Eigen::Vector2d& point0,
                         const Eigen::Vector2d& point1);

struct InitialisationModelOutcome {
  InitialisationModelSelection selection =
      InitialisationModelSelection::None;
  bool successful = false;
  size_t selectedInliers = 0;
};

InitialisationModelOutcome classifyInitialisationModels(
    size_t correspondences, bool rotationModelComputed,
    size_t rotationInliers, bool relativePoseModelComputed,
    size_t relativePoseInliers);

struct RansacOutcome {
  RansacStatus status = RansacStatus::NoPriorFrame;
  bool thresholdSuccess = false;
  bool returnedSuccess = false;
};

RansacOutcome classifyRansacOutcome(bool hasPriorFrame,
                                    size_t correspondences, size_t inliers,
                                    bool modelComputed);

}  // namespace diagnostics
}  // namespace okvis

#endif  // OKVIS_FRONTENDDIAGNOSTICS_HPP
