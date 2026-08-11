#include <cmath>
#include <limits>
#include <vector>

#include <gtest/gtest.h>
#include <Eigen/Core>
#include <opencv2/core/types.hpp>

#include <okvis/FrontendDiagnostics.hpp>

namespace {

using okvis::diagnostics::CameraDetectionAccumulator;
using okvis::diagnostics::CameraMapMatchAccumulator;
using okvis::diagnostics::DiagnosticDistribution;
using okvis::diagnostics::FrontendDiagnosticFrames;
using okvis::diagnostics::TriangulationAccumulator;
using okvis::diagnostics::TriangulationSource;

TEST(DiagnosticDistribution, UsesLinearInterpolatedQuantiles) {
  DiagnosticDistribution values;
  for (const double value : {1.0, 2.0, 3.0, 4.0}) {
    values.add(value);
  }
  values.add(std::numeric_limits<double>::quiet_NaN());

  const auto summary = values.summary();
  ASSERT_TRUE(summary.p10.has_value());
  ASSERT_TRUE(summary.median.has_value());
  ASSERT_TRUE(summary.p90.has_value());
  EXPECT_DOUBLE_EQ(*summary.p10, 1.3);
  EXPECT_DOUBLE_EQ(*summary.median, 2.5);
  EXPECT_DOUBLE_EQ(*summary.p90, 3.7);
}

TEST(DiagnosticDistribution, EmptyDistributionHasNoQuantiles) {
  const auto summary = DiagnosticDistribution{}.summary();
  EXPECT_FALSE(summary.p10.has_value());
  EXPECT_FALSE(summary.median.has_value());
  EXPECT_FALSE(summary.p90.has_value());
}

TEST(TriangulationAccumulator, MergePreservesCountsAndSamples) {
  TriangulationAccumulator a(TriangulationSource::SpatialStereo, 0, 1);
  TriangulationAccumulator b(TriangulationSource::SpatialStereo, 0, 1);
  a.recordAttempt(0.10, 0.02, 4.0, 3.0, true, false);
  b.recordAttempt(0.12, 0.03, 5.0, 4.0, true, true);
  a.recordDescriptorCandidate();
  b.recordEpipolarRejected();
  b.recordLandmarkBirth();

  a.merge(b);

  EXPECT_EQ(a.attempts(), 2);
  EXPECT_EQ(a.parallel(), 1);
  const auto record = a.toRecord(10, 20);
  EXPECT_EQ(record.descriptorCandidates, 1);
  EXPECT_EQ(record.epipolarRejected, 1);
  EXPECT_EQ(record.landmarkBirths, 1);
  ASSERT_TRUE(record.baselineM.median.has_value());
  EXPECT_DOUBLE_EQ(*record.baselineM.median, 0.11);
}

TEST(CameraMapMatchAccumulator, MergeKeepsRejectedFunnelWithoutMatches) {
  CameraMapMatchAccumulator destination;
  CameraMapMatchAccumulator rejected;
  rejected.projectedEligible = 7;
  rejected.descriptorComparisons = 21;
  rejected.descriptorCandidatesBelowThreshold = 4;
  rejected.epipolarRejected = 3;
  rejected.divergentRayRejected = 1;
  rejected.bestDescriptorDistance.add(12.0);

  destination.merge(rejected);

  EXPECT_EQ(destination.projectedEligible, 7);
  EXPECT_EQ(destination.descriptorComparisons, 21);
  EXPECT_EQ(destination.descriptorCandidatesBelowThreshold, 4);
  EXPECT_EQ(destination.epipolarRejected, 3);
  EXPECT_EQ(destination.divergentRayRejected, 1);
  EXPECT_EQ(destination.acceptedInitialised, 0);
  EXPECT_EQ(destination.acceptedUninitialised, 0);
  EXPECT_DOUBLE_EQ(*destination.bestDescriptorDistance.summary().median, 12.0);
}

TEST(KeypointCoverage, HandlesEmptyAndSinglePoint) {
  const auto empty = okvis::diagnostics::summarizeKeypoints({}, 100, 50);
  EXPECT_EQ(empty.keypoints, 0);
  EXPECT_DOUBLE_EQ(empty.occupiedGridFraction, 0.0);
  EXPECT_DOUBLE_EQ(empty.convexHullFraction, 0.0);

  const auto single = okvis::diagnostics::summarizeKeypoints(
      {cv::KeyPoint(20.0F, 10.0F, 1.0F)}, 100, 50);
  EXPECT_EQ(single.keypoints, 1);
  EXPECT_DOUBLE_EQ(single.occupiedGridFraction, 1.0 / 16.0);
  EXPECT_DOUBLE_EQ(single.convexHullFraction, 0.0);
}

TEST(KeypointCoverage, FourImageCornersCoverGridAndHull) {
  const std::vector<cv::KeyPoint> keypoints = {
      cv::KeyPoint(0.0F, 0.0F, 1.0F),
      cv::KeyPoint(100.0F, 0.0F, 1.0F),
      cv::KeyPoint(100.0F, 50.0F, 1.0F),
      cv::KeyPoint(0.0F, 50.0F, 1.0F)};

  const auto summary =
      okvis::diagnostics::summarizeKeypoints(keypoints, 100, 50);

  EXPECT_DOUBLE_EQ(summary.occupiedGridFraction, 4.0 / 16.0);
  EXPECT_DOUBLE_EQ(summary.convexHullFraction, 1.0);
}

TEST(KeypointCoverage, ClampsOutOfBoundsOnlyForGridIndexing) {
  const std::vector<cv::KeyPoint> keypoints = {
      cv::KeyPoint(-10.0F, -10.0F, 1.0F),
      cv::KeyPoint(110.0F, -10.0F, 1.0F),
      cv::KeyPoint(110.0F, 60.0F, 1.0F),
      cv::KeyPoint(-10.0F, 60.0F, 1.0F)};

  const auto summary =
      okvis::diagnostics::summarizeKeypoints(keypoints, 100, 50);

  EXPECT_DOUBLE_EQ(summary.occupiedGridFraction, 4.0 / 16.0);
  EXPECT_DOUBLE_EQ(summary.convexHullFraction, 1.0);
}

TEST(FrontendDiagnosticFrames, BindsDetectionByTimestampAndTakesOnce) {
  FrontendDiagnosticFrames frames(4);
  CameraDetectionAccumulator cameraSummary;
  cameraSummary.keypoints = 12;
  frames.updateDetection(1000, 2, cameraSummary);
  frames.updateDetection(1000, 0, cameraSummary);

  const auto bound = frames.bindFrame(7, 1000);
  ASSERT_TRUE(bound);
  EXPECT_EQ(bound->frameId, 7);
  EXPECT_EQ(bound->cameras.at(0).keypoints, 12);
  EXPECT_EQ(bound->cameras.at(1).keypoints, 0);
  EXPECT_EQ(bound->cameras.at(2).keypoints, 12);
  EXPECT_EQ(bound->mapMatching.size(), 4);

  auto completed = frames.take(7);
  ASSERT_TRUE(completed.has_value());
  EXPECT_EQ(completed->timestampNs, 1000);
  EXPECT_EQ(completed->cameras.size(), 4);
  EXPECT_FALSE(frames.take(7).has_value());
}

TEST(FrontendDiagnosticFrames, RepeatedBindReturnsSameAccumulator) {
  FrontendDiagnosticFrames frames(2);
  const auto first = frames.bindFrame(4, 2000);
  const auto second = frames.bindFrame(4, 2000);
  EXPECT_EQ(first, second);
}

TEST(DiagnosticGeometry, ComputesPhysicalTriangulationQuantities) {
  EXPECT_NEAR(okvis::diagnostics::cameraBaseline(
                  Eigen::Vector3d::Zero(), Eigen::Vector3d(0.1, 0.0, 0.0)),
              0.1, 1e-12);
  const auto angle = okvis::diagnostics::rayAngle(
      Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY());
  ASSERT_TRUE(angle.has_value());
  EXPECT_NEAR(*angle, 0.5 * std::acos(-1.0), 1e-12);
  EXPECT_FALSE(okvis::diagnostics::rayAngle(
                   Eigen::Vector3d::Zero(), Eigen::Vector3d::UnitY())
                   .has_value());
  EXPECT_NEAR(okvis::diagnostics::pixelDisplacement(
                  Eigen::Vector2d(1.0, 2.0), Eigen::Vector2d(4.0, 6.0)),
              5.0, 1e-12);
}

TEST(InitialisationModelClassifier, DistinguishesRotationRelativeAndFailure) {
  const auto rotation = okvis::diagnostics::classifyInitialisationModels(
      30, true, 27, true, 18);
  EXPECT_EQ(rotation.selection,
            okvis::diagnostics::InitialisationModelSelection::RotationOnly);
  EXPECT_TRUE(rotation.successful);
  EXPECT_EQ(rotation.selectedInliers, 27);

  const auto relative = okvis::diagnostics::classifyInitialisationModels(
      30, true, 15, true, 27);
  EXPECT_EQ(relative.selection,
            okvis::diagnostics::InitialisationModelSelection::RelativePose);
  EXPECT_TRUE(relative.successful);
  EXPECT_EQ(relative.selectedInliers, 27);

  const auto failed = okvis::diagnostics::classifyInitialisationModels(
      30, false, 0, false, 0);
  EXPECT_EQ(failed.selection,
            okvis::diagnostics::InitialisationModelSelection::None);
  EXPECT_FALSE(failed.successful);

  const auto insufficient = okvis::diagnostics::classifyInitialisationModels(
      7, false, 0, false, 0);
  EXPECT_EQ(insufficient.selection,
            okvis::diagnostics::InitialisationModelSelection::
                InsufficientCorrespondences);
}

TEST(RansacOutcomeClassifier, PreservesCurrentEarlyReturnSemantics) {
  const auto noPrior =
      okvis::diagnostics::classifyRansacOutcome(false, 0, 0, false);
  EXPECT_EQ(noPrior.status, okvis::diagnostics::RansacStatus::NoPriorFrame);
  EXPECT_FALSE(noPrior.returnedSuccess);

  const auto insufficient =
      okvis::diagnostics::classifyRansacOutcome(true, 7, 0, false);
  EXPECT_EQ(insufficient.status,
            okvis::diagnostics::RansacStatus::InsufficientCorrespondences);
  EXPECT_FALSE(insufficient.thresholdSuccess);
  EXPECT_TRUE(insufficient.returnedSuccess);

  const auto rejected =
      okvis::diagnostics::classifyRansacOutcome(true, 20, 9, true);
  EXPECT_EQ(rejected.status,
            okvis::diagnostics::RansacStatus::ThresholdRejected);
  EXPECT_FALSE(rejected.returnedSuccess);

  const auto accepted =
      okvis::diagnostics::classifyRansacOutcome(true, 20, 15, true);
  EXPECT_EQ(accepted.status,
            okvis::diagnostics::RansacStatus::ThresholdAccepted);
  EXPECT_TRUE(accepted.returnedSuccess);
}

}  // namespace
