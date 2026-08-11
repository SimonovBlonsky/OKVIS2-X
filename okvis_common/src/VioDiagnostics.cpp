#include <okvis/VioDiagnostics.hpp>

#include <cerrno>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <utility>

#include <fcntl.h>
#include <glog/logging.h>
#include <unistd.h>

namespace okvis {
namespace diagnostics {
namespace {

constexpr const char* kMetadataFilename = "vio_diag_metadata.csv";
constexpr const char* kFrameFilename = "vio_diag_frame.csv";
constexpr const char* kTriangulationFilename = "vio_diag_triangulation.csv";
constexpr const char* kInitialisationFilename = "vio_diag_initialisation.csv";
constexpr const char* kRansacFilename = "vio_diag_ransac.csv";
constexpr const char* kLandmarkEventsFilename = "vio_diag_landmark_events.csv";
constexpr const char* kActiveSentinel = ".vio_diagnostics.active";
constexpr const char* kCompleteSentinel = ".vio_diagnostics.complete";

const std::array<const char*, 6> kCsvFilenames = {
    kMetadataFilename, kFrameFilename, kTriangulationFilename,
    kInitialisationFilename, kRansacFilename, kLandmarkEventsFilename};

void createExclusiveEmptyFile(const std::filesystem::path& path) {
  const int descriptor = ::open(
      path.c_str(), O_WRONLY | O_CREAT | O_EXCL,
      S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH);
  if (descriptor < 0) {
    throw std::runtime_error(
        "cannot create completion sentinel: " +
        std::string(std::strerror(errno)));
  }
  if (::close(descriptor) != 0) {
    const int closeError = errno;
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    throw std::runtime_error(
        "cannot close completion sentinel: " +
        std::string(std::strerror(closeError)));
  }
}

std::string csvEscape(const std::string& value) {
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped.push_back('"');
  for (const char character : value) {
    if (character == '"') {
      escaped.push_back('"');
    }
    escaped.push_back(character);
  }
  escaped.push_back('"');
  return escaped;
}

std::string finiteDouble(const double value) {
  if (!std::isfinite(value)) {
    return {};
  }
  std::ostringstream stream;
  stream.imbue(std::locale::classic());
  stream << std::setprecision(17) << value;
  return stream.str();
}

std::string optionalDouble(const std::optional<double>& value) {
  return value ? finiteDouble(*value) : std::string{};
}

std::string optionalUnsigned(const std::optional<uint64_t>& value) {
  return value ? std::to_string(*value) : std::string{};
}

const char* toString(const TriangulationSource source) {
  switch (source) {
    case TriangulationSource::TemporalMotionStereo:
      return "temporal_motion_stereo";
    case TriangulationSource::SpatialStereo:
      return "spatial_stereo";
    case TriangulationSource::UninitialisedLandmark:
      return "uninitialised_landmark";
  }
  return "unknown";
}

const char* toString(const InitialisationModelSelection selection) {
  switch (selection) {
    case InitialisationModelSelection::InsufficientCorrespondences:
      return "insufficient_correspondences";
    case InitialisationModelSelection::RotationOnly:
      return "rotation_only";
    case InitialisationModelSelection::RelativePose:
      return "relative_pose";
    case InitialisationModelSelection::None:
      return "none";
  }
  return "unknown";
}

const char* toString(const RansacTrigger trigger) {
  switch (trigger) {
    case RansacTrigger::NoImu:
      return "no_imu";
    case RansacTrigger::LargeReprojectionError:
      return "large_reprojection_error";
    case RansacTrigger::TooFewAcceptedMatches:
      return "too_few_accepted_matches";
    case RansacTrigger::RetryWithUninitialisedLandmarks:
      return "retry_with_uninitialised_landmarks";
  }
  return "unknown";
}

const char* toString(const RansacStatus status) {
  switch (status) {
    case RansacStatus::NoPriorFrame:
      return "no_prior_frame";
    case RansacStatus::InsufficientCorrespondences:
      return "insufficient_correspondences";
    case RansacStatus::ModelComputationFailed:
      return "model_computation_failed";
    case RansacStatus::ThresholdRejected:
      return "threshold_rejected";
    case RansacStatus::ThresholdAccepted:
      return "threshold_accepted";
  }
  return "unknown";
}

const char* toString(const PoseSource source) {
  switch (source) {
    case PoseSource::DataAssociationEntryAfterAddStates:
      return "data_association_entry_after_add_states";
    case PoseSource::ImmediatePreInvocation:
      return "immediate_pre_invocation";
    case PoseSource::Gp3pModel:
      return "gp3p_model";
  }
  return "unknown";
}

const char* toString(const GraphRole role) {
  switch (role) {
    case GraphRole::Realtime:
      return "realtime";
    case GraphRole::Full:
      return "full";
  }
  return "unknown";
}

const char* toString(const LandmarkEventType eventType) {
  switch (eventType) {
    case LandmarkEventType::Birth:
      return "birth";
    case LandmarkEventType::Initialised:
      return "initialised";
    case LandmarkEventType::Deinitialised:
      return "deinitialised";
    case LandmarkEventType::ObservationAdded:
      return "observation_added";
    case LandmarkEventType::ObservationRemoved:
      return "observation_removed";
    case LandmarkEventType::LandmarkRemoved:
      return "landmark_removed";
    case LandmarkEventType::LandmarkMerged:
      return "landmark_merged";
  }
  return "unknown";
}

const char* toString(const RemovalReason reason) {
  switch (reason) {
    case RemovalReason::Gp3pOutlier:
      return "gp3p_outlier";
    case RemovalReason::PostOptimisationReprojection:
      return "post_optimisation_reprojection";
    case RemovalReason::Initialisation2d2dOutlier:
      return "initialisation_2d2d_outlier";
    case RemovalReason::LoopClosureReassociation:
      return "loop_closure_reassociation";
    case RemovalReason::StateMarginalisation:
      return "state_marginalisation";
    case RemovalReason::PoseGraphConversion:
      return "pose_graph_conversion";
    case RemovalReason::RealtimeFullGraphSync:
      return "realtime_full_graph_sync";
    case RemovalReason::ExplicitLandmarkMerge:
      return "explicit_landmark_merge";
    case RemovalReason::UnobservedLandmarkCleanup:
      return "unobserved_landmark_cleanup";
    case RemovalReason::Unknown:
      return "unknown";
  }
  return "unknown";
}

template <typename Value>
void appendField(std::ostream& stream, bool& first, const Value& value) {
  if (!first) {
    stream << ',';
  }
  first = false;
  stream << value;
}

void appendField(std::ostream& stream, bool& first, const std::string& value) {
  appendField<std::string>(stream, first, csvEscape(value));
}

void appendField(std::ostream& stream, bool& first, const char* value) {
  appendField(stream, first, std::string(value));
}

void appendDistribution(std::ostream& stream, bool& first,
                        const DistributionSummary& distribution) {
  appendField(stream, first, optionalDouble(distribution.p10));
  appendField(stream, first, optionalDouble(distribution.median));
  appendField(stream, first, optionalDouble(distribution.p90));
}

void appendPose(std::ostream& stream, bool& first, const PoseSnapshot& pose) {
  appendField(stream, first, finiteDouble(pose.tx));
  appendField(stream, first, finiteDouble(pose.ty));
  appendField(stream, first, finiteDouble(pose.tz));
  appendField(stream, first, finiteDouble(pose.qw));
  appendField(stream, first, finiteDouble(pose.qx));
  appendField(stream, first, finiteDouble(pose.qy));
  appendField(stream, first, finiteDouble(pose.qz));
}

void appendOptionalPose(std::ostream& stream, bool& first,
                        const std::optional<PoseSnapshot>& pose) {
  if (pose) {
    appendPose(stream, first, *pose);
    return;
  }
  for (size_t index = 0; index < 7; ++index) {
    appendField(stream, first, std::string{});
  }
}

template <typename Value>
Value vectorValue(const std::vector<Value>& values, const size_t index,
                  const Value& fallback = Value{}) {
  return index < values.size() ? values[index] : fallback;
}

DistributionSummary distributionValue(
    const std::vector<DistributionSummary>& values, const size_t index) {
  return index < values.size() ? values[index] : DistributionSummary{};
}

}  // namespace

struct VioDiagnostics::Impl {
  explicit Impl(std::string directory)
      : outputDirectory(std::move(directory)),
        observationAdds(std::getenv("OKVIS_DIAGNOSTICS_OBSERVATION_ADDS") !=
                            nullptr &&
                        std::string(std::getenv(
                            "OKVIS_DIAGNOSTICS_OBSERVATION_ADDS")) == "1") {}

  ~Impl() { closeStreams(); }

  void fail(const std::string& message) noexcept {
    if (!failureLogged) {
      LOG(ERROR) << "VIO diagnostics disabled: " << message;
      failureLogged = true;
    }
    failure = true;
    active = false;
    closeStreams();
  }

  void closeStreams() noexcept {
    metadata.close();
    frame.close();
    triangulation.close();
    initialisation.close();
    ransac.close();
    landmarkEvents.close();
  }

  bool streamOk(std::ofstream& stream, const char* filename) noexcept {
    stream << '\n';
    stream.flush();
    if (!stream.good()) {
      fail(std::string("failed writing ") + filename);
      return false;
    }
    return true;
  }

  void openStream(std::ofstream& stream, const char* filename) {
    stream.open(outputDirectory / filename, std::ios::out | std::ios::trunc);
    stream.imbue(std::locale::classic());
    stream << std::setprecision(17);
    if (!stream.is_open() || !stream.good()) {
      throw std::runtime_error(std::string("failed opening ") + filename);
    }
  }

  void writeMetadataUnlocked(const std::string& key,
                             const std::string& value) noexcept {
    if (!active || failure) {
      return;
    }
    bool first = true;
    appendField(metadata, first, VioDiagnostics::kSchemaVersion);
    appendField(metadata, first, key);
    appendField(metadata, first, value);
    streamOk(metadata, kMetadataFilename);
  }

  std::filesystem::path outputDirectory;
  const bool observationAdds;
  mutable std::mutex mutex;
  bool configured = false;
  bool active = false;
  bool failure = false;
  bool failureLogged = false;
  bool finished = false;
  size_t cameraCount = 0;
  uint64_t nextEventSequence = 1;
  std::ofstream metadata;
  std::ofstream frame;
  std::ofstream triangulation;
  std::ofstream initialisation;
  std::ofstream ransac;
  std::ofstream landmarkEvents;
};

VioDiagnostics& VioDiagnostics::instance() {
  const char* directory = std::getenv("OKVIS_DIAGNOSTICS_DIR");
  static VioDiagnostics diagnostics(directory == nullptr ? "" : directory);
  return diagnostics;
}

VioDiagnostics::VioDiagnostics(std::string outputDirectory)
    : impl_(new Impl(std::move(outputDirectory))) {}

VioDiagnostics::~VioDiagnostics() = default;

bool VioDiagnostics::configure(
    const size_t cameraCount,
    const std::map<std::string, std::string>& metadataValues) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->configured) {
    return impl_->active && !impl_->failure &&
           impl_->cameraCount == cameraCount;
  }
  impl_->configured = true;
  impl_->cameraCount = cameraCount;
  if (impl_->outputDirectory.empty()) {
    return false;
  }

  try {
    std::error_code error;
    std::filesystem::create_directories(impl_->outputDirectory, error);
    if (error) {
      throw std::runtime_error("cannot create output directory: " +
                               error.message());
    }
    for (const char* filename : kCsvFilenames) {
      if (std::filesystem::exists(impl_->outputDirectory / filename)) {
        throw std::runtime_error(std::string("output already exists: ") +
                                 filename);
      }
    }
    if (std::filesystem::exists(impl_->outputDirectory / kActiveSentinel) ||
        std::filesystem::exists(impl_->outputDirectory / kCompleteSentinel)) {
      throw std::runtime_error("diagnostics sentinel already exists");
    }
    if (!std::filesystem::create_directory(
            impl_->outputDirectory / kActiveSentinel, error) || error) {
      throw std::runtime_error("cannot acquire output directory: " +
                               error.message());
    }

    impl_->openStream(impl_->metadata, kMetadataFilename);
    impl_->openStream(impl_->frame, kFrameFilename);
    impl_->openStream(impl_->triangulation, kTriangulationFilename);
    impl_->openStream(impl_->initialisation, kInitialisationFilename);
    impl_->openStream(impl_->ransac, kRansacFilename);
    impl_->openStream(impl_->landmarkEvents, kLandmarkEventsFilename);
    impl_->active = true;

    impl_->metadata << "schema_version,key,value\n";

    impl_->frame
        << "schema_version,timestamp_ns,frame_id,initialised,"
           "data_association_succeeded,tracking_quality_below_threshold,"
           "keyframe";
    for (size_t camera = 0; camera < cameraCount; ++camera) {
      impl_->frame << ",keypoints_cam" << camera << ",response_p10_cam"
                   << camera << ",response_median_cam" << camera
                   << ",response_p90_cam" << camera
                   << ",grid_fraction_cam" << camera
                   << ",hull_fraction_cam" << camera
                   << ",projected_eligible_cam" << camera
                   << ",descriptor_comparisons_cam" << camera
                   << ",descriptor_candidates_below_threshold_cam" << camera
                   << ",epipolar_rejected_cam" << camera
                   << ",divergent_ray_rejected_cam" << camera
                   << ",accepted_initialised_cam" << camera
                   << ",accepted_uninitialised_cam" << camera
                   << ",best_map_descriptor_distance_p10_cam" << camera
                   << ",best_map_descriptor_distance_median_cam" << camera
                   << ",best_map_descriptor_distance_p90_cam" << camera;
    }
    impl_->frame
        << ",loop_closure_map_matches,accepted_descriptor_distance_p10,"
           "accepted_descriptor_distance_median,"
           "accepted_descriptor_distance_p90,"
           "predicted_reprojection_error_px_p10,"
           "predicted_reprojection_error_px_median,"
           "predicted_reprojection_error_px_p90,tracking_quality,"
           "active_initialised_landmarks,active_uninitialised_landmarks,"
           "landmark_births,landmark_initialisations,observations_added";
    for (size_t reason = 0; reason < kRemovalReasonCount; ++reason) {
      impl_->frame << ",observations_removed_reason_" << reason;
    }
    impl_->frame << ",motion_stereo_matches\n";

    impl_->triangulation
        << "schema_version,timestamp_ns,frame_id,source,camera0,camera1,"
           "attempts,descriptor_candidates,valid,invalid,parallel,"
           "initialisable,back_projection_rejected,descriptor_rejected,"
           "epipolar_rejected,divergent_rays_rejected,depth_rejected,"
           "projection_rejected,reprojection_rejected,landmark_births,"
           "landmark_initialisations,baseline_m_p10,baseline_m_median,"
           "baseline_m_p90,ray_angle_rad_p10,ray_angle_rad_median,"
           "ray_angle_rad_p90,pixel_displacement_px_p10,"
           "pixel_displacement_px_median,pixel_displacement_px_p90,"
           "depth_m_p10,depth_m_median,depth_m_p90\n";

    impl_->initialisation
        << "schema_version,timestamp_ns,current_frame_id,older_frame_id,"
           "camera,invocation,correspondences,rotation_model_computed,"
           "rotation_inliers,rotation_inlier_ratio,"
           "relative_pose_model_computed,relative_pose_inliers,"
           "relative_pose_inlier_ratio,selected_model,"
           "selected_model_successful,selected_inliers,"
           "function_returned_success,function_return_value\n";

    impl_->ransac
        << "schema_version,timestamp_ns,frame_id,invocation,primary_trigger,"
           "trigger_mask,status,correspondences,inliers,outliers,"
           "removed_observations,inlier_ratio,model_computed,"
           "threshold_success,returned_success";
    for (size_t camera = 0; camera < cameraCount; ++camera) {
      impl_->ransac << ",correspondences_cam" << camera << ",inliers_cam"
                    << camera << ",correspondence_grid_fraction_cam" << camera
                    << ",inlier_grid_fraction_cam" << camera;
    }
    impl_->ransac
        << ",data_association_start_pose_source,"
           "pre_invocation_pose_source,data_association_start_tx,"
           "data_association_start_ty,data_association_start_tz,"
           "data_association_start_qw,data_association_start_qx,"
           "data_association_start_qy,data_association_start_qz,"
           "pre_invocation_tx,pre_invocation_ty,pre_invocation_tz,"
           "pre_invocation_qw,pre_invocation_qx,pre_invocation_qy,"
           "pre_invocation_qz,gp3p_model_tx,gp3p_model_ty,gp3p_model_tz,"
           "gp3p_model_qw,gp3p_model_qx,gp3p_model_qy,gp3p_model_qz,"
           "start_to_model_rotation_rad,"
           "start_to_model_translation_m,"
           "pre_invocation_to_model_rotation_rad,"
           "pre_invocation_to_model_translation_m\n";

    impl_->landmarkEvents
        << "schema_version,event_sequence,event_timestamp_ns,event_frame_id,"
           "subject_timestamp_ns,subject_frame_id,birth_timestamp_ns,"
           "birth_frame_id,landmark_id,graph_role,event_type,reason,"
           "initialised_before,initialised_after,observations_before,"
           "observations_after,quality\n";

    if (!impl_->metadata.good() || !impl_->frame.good() ||
        !impl_->triangulation.good() || !impl_->initialisation.good() ||
        !impl_->ransac.good() || !impl_->landmarkEvents.good()) {
      throw std::runtime_error("failed writing CSV headers");
    }

    impl_->writeMetadataUnlocked("schema_version",
                                 std::to_string(kSchemaVersion));
    impl_->writeMetadataUnlocked("camera_count", std::to_string(cameraCount));
    impl_->writeMetadataUnlocked("run_complete", "false");
    impl_->writeMetadataUnlocked(
        "observation_adds_enabled",
        impl_->observationAdds ? "true" : "false");
    for (const auto& metadataValue : metadataValues) {
      impl_->writeMetadataUnlocked(metadataValue.first, metadataValue.second);
    }
    return impl_->active && !impl_->failure;
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown configuration error");
  }
  return false;
}

bool VioDiagnostics::enabled() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->active && !impl_->failure;
}

bool VioDiagnostics::failed() const {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  return impl_->failure;
}

bool VioDiagnostics::observationAddsEnabled() const {
  return impl_->observationAdds;
}

void VioDiagnostics::writeMetadata(const std::string& key,
                                   const std::string& value) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->writeMetadataUnlocked(key, value);
}

void VioDiagnostics::writeFrame(const FrameDiagnosticRecord& record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->active || impl_->failure) {
    return;
  }
  try {
    bool first = true;
    appendField(impl_->frame, first, kSchemaVersion);
    appendField(impl_->frame, first, record.timestampNs);
    appendField(impl_->frame, first, record.frameId);
    appendField(impl_->frame, first, record.initialised ? 1 : 0);
    appendField(impl_->frame, first,
                record.dataAssociationSucceeded ? 1 : 0);
    appendField(impl_->frame, first,
                record.trackingQualityBelowThreshold ? 1 : 0);
    appendField(impl_->frame, first, record.keyframe ? 1 : 0);
    for (size_t camera = 0; camera < impl_->cameraCount; ++camera) {
      appendField(impl_->frame, first,
                  vectorValue(record.keypointCount, camera));
      appendDistribution(impl_->frame, first,
                         distributionValue(record.keypointResponse, camera));
      appendField(impl_->frame, first,
                  finiteDouble(vectorValue(record.occupiedGridFraction,
                                           camera,
                                           std::numeric_limits<double>::quiet_NaN())));
      appendField(impl_->frame, first,
                  finiteDouble(vectorValue(record.convexHullFraction, camera,
                                           std::numeric_limits<double>::quiet_NaN())));
      appendField(impl_->frame, first,
                  vectorValue(record.projectedEligibleMapLandmarks, camera));
      appendField(impl_->frame, first,
                  vectorValue(record.mapDescriptorComparisons, camera));
      appendField(impl_->frame, first,
                  vectorValue(record.mapDescriptorCandidatesBelowThreshold,
                              camera));
      appendField(impl_->frame, first,
                  vectorValue(record.mapEpipolarRejected, camera));
      appendField(impl_->frame, first,
                  vectorValue(record.mapDivergentRayRejected, camera));
      appendField(impl_->frame, first,
                  vectorValue(record.acceptedInitialisedMapMatches, camera));
      appendField(impl_->frame, first,
                  vectorValue(record.acceptedUninitialisedMapMatches, camera));
      appendDistribution(impl_->frame, first,
                         distributionValue(record.bestMapDescriptorDistance,
                                           camera));
    }
    appendField(impl_->frame, first, record.loopClosureMapMatches);
    appendDistribution(impl_->frame, first,
                       record.acceptedDescriptorDistance);
    appendDistribution(impl_->frame, first,
                       record.predictedReprojectionErrorPx);
    appendField(impl_->frame, first, optionalDouble(record.trackingQuality));
    appendField(impl_->frame, first, record.activeInitialisedLandmarks);
    appendField(impl_->frame, first, record.activeUninitialisedLandmarks);
    appendField(impl_->frame, first, record.landmarkBirths);
    appendField(impl_->frame, first, record.landmarkInitialisations);
    appendField(impl_->frame, first, record.observationsAdded);
    for (const size_t count : record.observationsRemovedByReason) {
      appendField(impl_->frame, first, count);
    }
    appendField(impl_->frame, first, record.motionStereoMatches);
    impl_->streamOk(impl_->frame, kFrameFilename);
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown frame write error");
  }
}

void VioDiagnostics::writeTriangulation(
    const TriangulationDiagnosticRecord& record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->active || impl_->failure) {
    return;
  }
  try {
    bool first = true;
    appendField(impl_->triangulation, first, kSchemaVersion);
    appendField(impl_->triangulation, first, record.timestampNs);
    appendField(impl_->triangulation, first, record.frameId);
    appendField(impl_->triangulation, first, toString(record.source));
    appendField(impl_->triangulation, first, record.camera0);
    appendField(impl_->triangulation, first, record.camera1);
    appendField(impl_->triangulation, first, record.attempts);
    appendField(impl_->triangulation, first, record.descriptorCandidates);
    appendField(impl_->triangulation, first, record.valid);
    appendField(impl_->triangulation, first, record.invalid);
    appendField(impl_->triangulation, first, record.parallel);
    appendField(impl_->triangulation, first, record.initialisable);
    appendField(impl_->triangulation, first, record.backProjectionRejected);
    appendField(impl_->triangulation, first, record.descriptorRejected);
    appendField(impl_->triangulation, first, record.epipolarRejected);
    appendField(impl_->triangulation, first, record.divergentRaysRejected);
    appendField(impl_->triangulation, first, record.depthRejected);
    appendField(impl_->triangulation, first, record.projectionRejected);
    appendField(impl_->triangulation, first, record.reprojectionRejected);
    appendField(impl_->triangulation, first, record.landmarkBirths);
    appendField(impl_->triangulation, first, record.landmarkInitialisations);
    appendDistribution(impl_->triangulation, first, record.baselineM);
    appendDistribution(impl_->triangulation, first, record.rayAngleRad);
    appendDistribution(impl_->triangulation, first,
                       record.pixelDisplacementPx);
    appendDistribution(impl_->triangulation, first, record.depthM);
    impl_->streamOk(impl_->triangulation, kTriangulationFilename);
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown triangulation write error");
  }
}

void VioDiagnostics::writeInitialisation(
    const InitialisationDiagnosticRecord& record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->active || impl_->failure) {
    return;
  }
  try {
    bool first = true;
    appendField(impl_->initialisation, first, kSchemaVersion);
    appendField(impl_->initialisation, first, record.timestampNs);
    appendField(impl_->initialisation, first, record.currentFrameId);
    appendField(impl_->initialisation, first, record.olderFrameId);
    appendField(impl_->initialisation, first, record.camera);
    appendField(impl_->initialisation, first, record.invocation);
    appendField(impl_->initialisation, first, record.correspondences);
    appendField(impl_->initialisation, first,
                record.rotationModelComputed ? 1 : 0);
    appendField(impl_->initialisation, first, record.rotationInliers);
    appendField(impl_->initialisation, first,
                optionalDouble(record.rotationInlierRatio));
    appendField(impl_->initialisation, first,
                record.relativePoseModelComputed ? 1 : 0);
    appendField(impl_->initialisation, first, record.relativePoseInliers);
    appendField(impl_->initialisation, first,
                optionalDouble(record.relativePoseInlierRatio));
    appendField(impl_->initialisation, first, toString(record.selectedModel));
    appendField(impl_->initialisation, first,
                record.selectedModelSuccessful ? 1 : 0);
    appendField(impl_->initialisation, first, record.selectedInliers);
    appendField(impl_->initialisation, first,
                record.functionReturnedSuccess ? 1 : 0);
    appendField(impl_->initialisation, first, record.functionReturnValue);
    impl_->streamOk(impl_->initialisation, kInitialisationFilename);
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown initialisation write error");
  }
}

void VioDiagnostics::writeRansac(const RansacDiagnosticRecord& record) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->active || impl_->failure) {
    return;
  }
  try {
    bool first = true;
    appendField(impl_->ransac, first, kSchemaVersion);
    appendField(impl_->ransac, first, record.timestampNs);
    appendField(impl_->ransac, first, record.frameId);
    appendField(impl_->ransac, first, record.invocation);
    appendField(impl_->ransac, first, toString(record.primaryTrigger));
    appendField(impl_->ransac, first, record.triggerMask);
    appendField(impl_->ransac, first, toString(record.status));
    appendField(impl_->ransac, first, record.correspondences);
    appendField(impl_->ransac, first, record.inliers);
    appendField(impl_->ransac, first, record.outliers);
    appendField(impl_->ransac, first, record.removedObservations);
    appendField(impl_->ransac, first, optionalDouble(record.inlierRatio));
    appendField(impl_->ransac, first, record.modelComputed ? 1 : 0);
    appendField(impl_->ransac, first, record.thresholdSuccess ? 1 : 0);
    appendField(impl_->ransac, first, record.returnedSuccess ? 1 : 0);
    for (size_t camera = 0; camera < impl_->cameraCount; ++camera) {
      appendField(impl_->ransac, first,
                  vectorValue(record.correspondencesPerCamera, camera));
      appendField(impl_->ransac, first,
                  vectorValue(record.inliersPerCamera, camera));
      appendField(impl_->ransac, first,
                  finiteDouble(vectorValue(
                      record.correspondenceGridFractionPerCamera, camera,
                      std::numeric_limits<double>::quiet_NaN())));
      appendField(impl_->ransac, first,
                  finiteDouble(vectorValue(record.inlierGridFractionPerCamera,
                                           camera,
                                           std::numeric_limits<double>::quiet_NaN())));
    }
    appendField(impl_->ransac, first,
                toString(record.dataAssociationStartPoseSource));
    appendField(impl_->ransac, first,
                toString(record.preInvocationPoseSource));
    appendPose(impl_->ransac, first, record.dataAssociationStartPose);
    appendPose(impl_->ransac, first, record.preInvocationPose);
    appendOptionalPose(impl_->ransac, first, record.gp3pModelPose);
    appendField(impl_->ransac, first,
                optionalDouble(record.startToModelRotationRad));
    appendField(impl_->ransac, first,
                optionalDouble(record.startToModelTranslationM));
    appendField(impl_->ransac, first,
                optionalDouble(record.preInvocationToModelRotationRad));
    appendField(impl_->ransac, first,
                optionalDouble(record.preInvocationToModelTranslationM));
    impl_->streamOk(impl_->ransac, kRansacFilename);
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown RANSAC write error");
  }
}

void VioDiagnostics::writeLandmarkEvents(
    std::vector<LandmarkEventRecord> records) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->active || impl_->failure) {
    return;
  }
  try {
    for (LandmarkEventRecord& record : records) {
      record.eventSequence = impl_->nextEventSequence++;
      bool first = true;
      appendField(impl_->landmarkEvents, first, kSchemaVersion);
      appendField(impl_->landmarkEvents, first, record.eventSequence);
      appendField(impl_->landmarkEvents, first, record.eventTimestampNs);
      appendField(impl_->landmarkEvents, first, record.eventFrameId);
      appendField(impl_->landmarkEvents, first,
                  optionalUnsigned(record.subjectTimestampNs));
      appendField(impl_->landmarkEvents, first,
                  optionalUnsigned(record.subjectFrameId));
      appendField(impl_->landmarkEvents, first,
                  optionalUnsigned(record.birthTimestampNs));
      appendField(impl_->landmarkEvents, first,
                  optionalUnsigned(record.birthFrameId));
      appendField(impl_->landmarkEvents, first, record.landmarkId);
      appendField(impl_->landmarkEvents, first, toString(record.graphRole));
      appendField(impl_->landmarkEvents, first, toString(record.eventType));
      appendField(impl_->landmarkEvents, first, toString(record.reason));
      appendField(impl_->landmarkEvents, first,
                  record.initialisedBefore ? 1 : 0);
      appendField(impl_->landmarkEvents, first,
                  record.initialisedAfter ? 1 : 0);
      appendField(impl_->landmarkEvents, first, record.observationsBefore);
      appendField(impl_->landmarkEvents, first, record.observationsAfter);
      appendField(impl_->landmarkEvents, first,
                  optionalDouble(record.quality));
      if (!impl_->streamOk(impl_->landmarkEvents,
                           kLandmarkEventsFilename)) {
        break;
      }
    }
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown landmark-event write error");
  }
}

void VioDiagnostics::finish(const bool successful) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->finished) {
    return;
  }
  impl_->finished = true;
  if (!impl_->active || impl_->failure) {
    impl_->closeStreams();
    return;
  }
  try {
    impl_->writeMetadataUnlocked("run_complete",
                                 successful ? "true" : "false");
    if (impl_->failure) {
      return;
    }
    impl_->closeStreams();
    if (!successful) {
      impl_->active = false;
      return;
    }

    createExclusiveEmptyFile(impl_->outputDirectory / kCompleteSentinel);
    std::error_code error;
    if (!std::filesystem::remove(impl_->outputDirectory / kActiveSentinel,
                                 error) ||
        error) {
      throw std::runtime_error("cannot release output directory: " +
                               error.message());
    }
    impl_->active = false;
  } catch (const std::exception& exception) {
    impl_->fail(exception.what());
  } catch (...) {
    impl_->fail("unknown finalisation error");
  }
}

}  // namespace diagnostics
}  // namespace okvis
