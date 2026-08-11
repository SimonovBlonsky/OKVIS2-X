#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <okvis/VioDiagnostics.hpp>

namespace {

std::string readFile(const std::filesystem::path& path) {
  std::ifstream stream(path);
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

std::vector<std::string> splitLines(const std::string& value) {
  std::vector<std::string> lines;
  std::stringstream stream(value);
  std::string line;
  while (std::getline(stream, line)) {
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::string> splitFields(const std::string& value) {
  std::vector<std::string> fields;
  std::stringstream stream(value);
  std::string field;
  while (std::getline(stream, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

class VioDiagnosticsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::path(testing::TempDir()) /
            ("okvis_vio_diagnostics_" +
             std::to_string(reinterpret_cast<uintptr_t>(this)));
    std::filesystem::remove_all(root_);
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  std::filesystem::path root_;
};

TEST_F(VioDiagnosticsTest, EmptyDirectoryDisablesAllOutput) {
  okvis::diagnostics::VioDiagnostics writer("");

  EXPECT_FALSE(writer.configure(4));
  EXPECT_FALSE(writer.enabled());
  writer.writeFrame({});
  writer.writeTriangulation({});
  writer.writeInitialisation({});
  writer.writeRansac({});
  writer.writeLandmarkEvents({{}});
  writer.finish(true);

  EXPECT_FALSE(std::filesystem::exists(root_));
}

TEST_F(VioDiagnosticsTest, WritesSixCsvFilesAndCompletionSentinel) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  ASSERT_TRUE(writer.configure(2, {{"build_id", "test,quoted"}}));

  okvis::diagnostics::FrameDiagnosticRecord frame;
  frame.timestampNs = 10;
  frame.frameId = 20;
  frame.keypointCount = {3, 4};
  frame.keypointResponse.resize(2);
  frame.occupiedGridFraction = {0.25, 0.5};
  frame.convexHullFraction = {0.1, 0.2};
  frame.projectedEligibleMapLandmarks = {5, 6};
  frame.mapDescriptorComparisons = {7, 8};
  frame.mapDescriptorCandidatesBelowThreshold = {1, 2};
  frame.mapEpipolarRejected = {0, 1};
  frame.mapDivergentRayRejected = {0, 0};
  frame.acceptedInitialisedMapMatches = {1, 1};
  frame.acceptedUninitialisedMapMatches = {0, 1};
  frame.bestMapDescriptorDistance.resize(2);
  writer.writeFrame(frame);
  writer.writeTriangulation({});
  writer.writeInitialisation({});
  writer.writeRansac({});
  writer.writeLandmarkEvents({{}});
  writer.finish(true);

  for (const char* filename : {
           "vio_diag_metadata.csv",
           "vio_diag_frame.csv",
           "vio_diag_triangulation.csv",
           "vio_diag_initialisation.csv",
           "vio_diag_ransac.csv",
           "vio_diag_landmark_events.csv",
       }) {
    EXPECT_TRUE(std::filesystem::is_regular_file(root_ / filename)) << filename;
  }
  EXPECT_TRUE(std::filesystem::is_regular_file(
      root_ / ".vio_diagnostics.complete"));
  EXPECT_FALSE(std::filesystem::exists(root_ / ".vio_diagnostics.active"));

  const std::string frameCsv = readFile(root_ / "vio_diag_frame.csv");
  EXPECT_NE(frameCsv.find("keypoints_cam1"), std::string::npos);
  EXPECT_NE(frameCsv.find("descriptor_comparisons_cam1"), std::string::npos);
  EXPECT_NE(frameCsv.find(",,"), std::string::npos);
  EXPECT_NE(readFile(root_ / "vio_diag_metadata.csv")
                .find("build_id,\"test,quoted\""),
            std::string::npos);
}

TEST_F(VioDiagnosticsTest, WritesStableFrameAndRansacHeaders) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  ASSERT_TRUE(writer.configure(1));
  writer.finish(true);

  const auto frameLines = splitLines(readFile(root_ / "vio_diag_frame.csv"));
  ASSERT_FALSE(frameLines.empty());
  const auto frameFields = splitFields(frameLines.front());
  for (const char* required : {
           "response_p10_cam0",
           "response_median_cam0",
           "response_p90_cam0",
           "grid_fraction_cam0",
           "hull_fraction_cam0",
           "projected_eligible_cam0",
           "descriptor_comparisons_cam0",
           "descriptor_candidates_below_threshold_cam0",
           "epipolar_rejected_cam0",
           "divergent_ray_rejected_cam0",
           "accepted_initialised_cam0",
           "accepted_uninitialised_cam0",
       }) {
    EXPECT_NE(std::find(frameFields.begin(), frameFields.end(), required),
              frameFields.end())
        << required;
  }

  const auto ransacLines =
      splitLines(readFile(root_ / "vio_diag_ransac.csv"));
  ASSERT_FALSE(ransacLines.empty());
  const auto ransacFields = splitFields(ransacLines.front());
  for (const char* required : {
           "data_association_start_tx",
           "data_association_start_ty",
           "data_association_start_tz",
           "data_association_start_qw",
           "data_association_start_qx",
           "data_association_start_qy",
           "data_association_start_qz",
           "pre_invocation_tx",
           "pre_invocation_ty",
           "pre_invocation_tz",
           "pre_invocation_qw",
           "pre_invocation_qx",
           "pre_invocation_qy",
           "pre_invocation_qz",
           "gp3p_model_tx",
           "gp3p_model_ty",
           "gp3p_model_tz",
           "gp3p_model_qw",
           "gp3p_model_qx",
           "gp3p_model_qy",
           "gp3p_model_qz",
       }) {
    EXPECT_NE(std::find(ransacFields.begin(), ransacFields.end(), required),
              ransacFields.end())
        << required;
  }
}

TEST_F(VioDiagnosticsTest, AssignsStrictEventSequenceAcrossBatches) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  ASSERT_TRUE(writer.configure(1));

  okvis::diagnostics::LandmarkEventRecord first;
  first.landmarkId = 10;
  okvis::diagnostics::LandmarkEventRecord second;
  second.landmarkId = 11;
  writer.writeLandmarkEvents({first});
  writer.writeLandmarkEvents({second});
  writer.finish(true);

  const auto lines = splitLines(
      readFile(root_ / "vio_diag_landmark_events.csv"));
  ASSERT_EQ(lines.size(), 3);
  EXPECT_EQ(lines[1].substr(0, 4), "1,1,");
  EXPECT_EQ(lines[2].substr(0, 4), "1,2,");
}

TEST_F(VioDiagnosticsTest, RefusesExistingDiagnosticOutput) {
  std::filesystem::create_directories(root_);
  std::ofstream(root_ / "vio_diag_frame.csv") << "existing\n";

  okvis::diagnostics::VioDiagnostics writer(root_.string());
  EXPECT_FALSE(writer.configure(1));
  EXPECT_FALSE(writer.enabled());
  EXPECT_EQ(readFile(root_ / "vio_diag_frame.csv"), "existing\n");
}

TEST_F(VioDiagnosticsTest, ConfigureIsIdempotentOnlyForSameCameraCount) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  EXPECT_TRUE(writer.configure(4));
  EXPECT_TRUE(writer.configure(4));
  EXPECT_FALSE(writer.configure(3));
  writer.finish(false);
  EXPECT_FALSE(std::filesystem::exists(
      root_ / ".vio_diagnostics.complete"));
}

TEST_F(VioDiagnosticsTest, SuccessfulFinishIsIdempotent) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  ASSERT_TRUE(writer.configure(1));

  writer.finish(true);
  writer.finish(true);

  const auto metadataLines =
      splitLines(readFile(root_ / "vio_diag_metadata.csv"));
  EXPECT_EQ(std::count(metadataLines.begin(), metadataLines.end(),
                       "1,run_complete,true"),
            1);
}

TEST_F(VioDiagnosticsTest, CompletionRacePreservesActiveAndExistingSentinel) {
  okvis::diagnostics::VioDiagnostics writer(root_.string());
  ASSERT_TRUE(writer.configure(1));
  const auto complete = root_ / ".vio_diagnostics.complete";
  std::ofstream(complete) << "racing-writer\n";

  writer.finish(true);

  EXPECT_TRUE(writer.failed());
  EXPECT_TRUE(std::filesystem::is_directory(
      root_ / ".vio_diagnostics.active"));
  EXPECT_EQ(readFile(complete), "racing-writer\n");
}

}  // namespace
