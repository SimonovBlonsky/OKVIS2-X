#ifndef OKVIS_VIODIAGNOSTICS_HPP
#define OKVIS_VIODIAGNOSTICS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace okvis {
namespace diagnostics {

enum class TriangulationSource {
  TemporalMotionStereo,
  SpatialStereo,
  UninitialisedLandmark
};

enum class InitialisationModelSelection {
  InsufficientCorrespondences,
  RotationOnly,
  RelativePose,
  None
};

enum class RansacTrigger : uint32_t {
  NoImu = 1u << 0,
  LargeReprojectionError = 1u << 1,
  TooFewAcceptedMatches = 1u << 2,
  RetryWithUninitialisedLandmarks = 1u << 3
};

enum class RansacStatus {
  NoPriorFrame,
  InsufficientCorrespondences,
  ModelComputationFailed,
  ThresholdRejected,
  ThresholdAccepted
};

enum class PoseSource {
  DataAssociationEntryAfterAddStates,
  ImmediatePreInvocation,
  Gp3pModel
};

enum class GraphRole { Realtime, Full };

enum class LandmarkEventType {
  Birth,
  Initialised,
  Deinitialised,
  ObservationAdded,
  ObservationRemoved,
  LandmarkRemoved,
  LandmarkMerged
};

enum class RemovalReason {
  Gp3pOutlier,
  PostOptimisationReprojection,
  Initialisation2d2dOutlier,
  LoopClosureReassociation,
  StateMarginalisation,
  PoseGraphConversion,
  RealtimeFullGraphSync,
  ExplicitLandmarkMerge,
  UnobservedLandmarkCleanup,
  Unknown
};

constexpr size_t kRemovalReasonCount = 10;

struct DistributionSummary {
  std::optional<double> p10;
  std::optional<double> median;
  std::optional<double> p90;
};

struct PoseSnapshot {
  double tx = 0.0;
  double ty = 0.0;
  double tz = 0.0;
  double qw = 1.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;
};

struct EventContext {
  uint64_t eventTimestampNs = 0;
  uint64_t eventFrameId = 0;
};

struct FrameDiagnosticRecord {
  uint64_t timestampNs = 0;
  uint64_t frameId = 0;
  bool initialised = false;
  bool dataAssociationSucceeded = false;
  bool trackingQualityBelowThreshold = false;
  bool keyframe = false;
  std::vector<size_t> keypointCount;
  std::vector<DistributionSummary> keypointResponse;
  std::vector<double> occupiedGridFraction;
  std::vector<double> convexHullFraction;
  std::vector<size_t> projectedEligibleMapLandmarks;
  std::vector<size_t> mapDescriptorComparisons;
  std::vector<size_t> mapDescriptorCandidatesBelowThreshold;
  std::vector<size_t> mapEpipolarRejected;
  std::vector<size_t> mapDivergentRayRejected;
  std::vector<size_t> acceptedInitialisedMapMatches;
  std::vector<size_t> acceptedUninitialisedMapMatches;
  std::vector<DistributionSummary> bestMapDescriptorDistance;
  size_t loopClosureMapMatches = 0;
  DistributionSummary acceptedDescriptorDistance;
  DistributionSummary predictedReprojectionErrorPx;
  std::optional<double> trackingQuality;
  size_t activeInitialisedLandmarks = 0;
  size_t activeUninitialisedLandmarks = 0;
  size_t landmarkBirths = 0;
  size_t landmarkInitialisations = 0;
  size_t observationsAdded = 0;
  std::array<size_t, kRemovalReasonCount> observationsRemovedByReason{};
  size_t motionStereoMatches = 0;
};

struct TriangulationDiagnosticRecord {
  uint64_t timestampNs = 0;
  uint64_t frameId = 0;
  TriangulationSource source = TriangulationSource::TemporalMotionStereo;
  int camera0 = -1;
  int camera1 = -1;
  size_t attempts = 0;
  size_t descriptorCandidates = 0;
  size_t valid = 0;
  size_t invalid = 0;
  size_t parallel = 0;
  size_t initialisable = 0;
  size_t backProjectionRejected = 0;
  size_t descriptorRejected = 0;
  size_t epipolarRejected = 0;
  size_t divergentRaysRejected = 0;
  size_t depthRejected = 0;
  size_t projectionRejected = 0;
  size_t reprojectionRejected = 0;
  size_t landmarkBirths = 0;
  size_t landmarkInitialisations = 0;
  DistributionSummary baselineM;
  DistributionSummary rayAngleRad;
  DistributionSummary pixelDisplacementPx;
  DistributionSummary depthM;
};

struct InitialisationDiagnosticRecord {
  uint64_t timestampNs = 0;
  uint64_t currentFrameId = 0;
  uint64_t olderFrameId = 0;
  int camera = -1;
  size_t invocation = 0;
  size_t correspondences = 0;
  bool rotationModelComputed = false;
  size_t rotationInliers = 0;
  std::optional<double> rotationInlierRatio;
  bool relativePoseModelComputed = false;
  size_t relativePoseInliers = 0;
  std::optional<double> relativePoseInlierRatio;
  InitialisationModelSelection selectedModel =
      InitialisationModelSelection::None;
  bool selectedModelSuccessful = false;
  size_t selectedInliers = 0;
  bool functionReturnedSuccess = false;
  int functionReturnValue = -1;
};

struct RansacDiagnosticRecord {
  uint64_t timestampNs = 0;
  uint64_t frameId = 0;
  size_t invocation = 0;
  RansacTrigger primaryTrigger = RansacTrigger::LargeReprojectionError;
  uint32_t triggerMask = 0;
  RansacStatus status = RansacStatus::NoPriorFrame;
  size_t correspondences = 0;
  size_t inliers = 0;
  size_t outliers = 0;
  size_t removedObservations = 0;
  std::optional<double> inlierRatio;
  bool modelComputed = false;
  bool thresholdSuccess = false;
  bool returnedSuccess = false;
  std::vector<size_t> correspondencesPerCamera;
  std::vector<size_t> inliersPerCamera;
  std::vector<double> correspondenceGridFractionPerCamera;
  std::vector<double> inlierGridFractionPerCamera;
  PoseSource dataAssociationStartPoseSource =
      PoseSource::DataAssociationEntryAfterAddStates;
  PoseSource preInvocationPoseSource = PoseSource::ImmediatePreInvocation;
  PoseSnapshot dataAssociationStartPose;
  PoseSnapshot preInvocationPose;
  std::optional<PoseSnapshot> gp3pModelPose;
  std::optional<double> startToModelRotationRad;
  std::optional<double> startToModelTranslationM;
  std::optional<double> preInvocationToModelRotationRad;
  std::optional<double> preInvocationToModelTranslationM;
};

struct LandmarkEventRecord {
  uint64_t eventSequence = 0;
  uint64_t eventTimestampNs = 0;
  uint64_t eventFrameId = 0;
  std::optional<uint64_t> subjectTimestampNs;
  std::optional<uint64_t> subjectFrameId;
  std::optional<uint64_t> birthTimestampNs;
  std::optional<uint64_t> birthFrameId;
  uint64_t landmarkId = 0;
  GraphRole graphRole = GraphRole::Realtime;
  LandmarkEventType eventType = LandmarkEventType::Birth;
  RemovalReason reason = RemovalReason::Unknown;
  bool initialisedBefore = false;
  bool initialisedAfter = false;
  size_t observationsBefore = 0;
  size_t observationsAfter = 0;
  std::optional<double> quality;
};

class VioDiagnostics {
 public:
  static constexpr int kSchemaVersion = 1;

  static VioDiagnostics& instance();

  explicit VioDiagnostics(std::string outputDirectory);
  ~VioDiagnostics();

  VioDiagnostics(const VioDiagnostics&) = delete;
  VioDiagnostics& operator=(const VioDiagnostics&) = delete;

  bool configure(
      size_t cameraCount,
      const std::map<std::string, std::string>& metadata = {});
  bool enabled() const;
  bool failed() const;
  bool observationAddsEnabled() const;
  void writeMetadata(const std::string& key, const std::string& value);
  void writeFrame(const FrameDiagnosticRecord& record);
  void writeTriangulation(const TriangulationDiagnosticRecord& record);
  void writeInitialisation(const InitialisationDiagnosticRecord& record);
  void writeRansac(const RansacDiagnosticRecord& record);
  void writeLandmarkEvents(std::vector<LandmarkEventRecord> records);
  void finish(bool successful);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace diagnostics
}  // namespace okvis

#endif  // OKVIS_VIODIAGNOSTICS_HPP
