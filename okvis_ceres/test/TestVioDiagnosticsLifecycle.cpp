#include <gtest/gtest.h>

#include <okvis/ViGraph.hpp>

namespace {

TEST(VioDiagnosticsLifecycle, PreservesBirthContextAndRemovalReason) {
  okvis::ViGraph graph;
  graph.setDiagnosticsGraphRole(okvis::diagnostics::GraphRole::Realtime);
  graph.setDiagnosticsCollectionEnabled(true);

  const okvis::LandmarkId landmarkId(1);
  ASSERT_TRUE(graph.addLandmark(
      landmarkId, Eigen::Vector4d(1.0, 2.0, 3.0, 1.0), false));
  graph.registerLandmarkBirth(
      landmarkId, okvis::diagnostics::EventContext{100, 10}, true);
  auto birthEvents = graph.takeDiagnosticEvents();
  ASSERT_EQ(birthEvents.size(), 1);
  EXPECT_EQ(birthEvents.front().eventType,
            okvis::diagnostics::LandmarkEventType::Birth);

  ASSERT_TRUE(graph.removeLandmark(
      landmarkId,
      okvis::diagnostics::RemovalReason::StateMarginalisation,
      okvis::diagnostics::EventContext{500, 50}));
  auto events = graph.takeDiagnosticEvents();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().graphRole,
            okvis::diagnostics::GraphRole::Realtime);
  EXPECT_EQ(events.front().eventType,
            okvis::diagnostics::LandmarkEventType::LandmarkRemoved);
  EXPECT_EQ(events.front().reason,
            okvis::diagnostics::RemovalReason::StateMarginalisation);
  EXPECT_EQ(events.front().eventTimestampNs, 500);
  EXPECT_EQ(events.front().eventFrameId, 50);
  ASSERT_TRUE(events.front().birthTimestampNs.has_value());
  EXPECT_EQ(*events.front().birthTimestampNs, 100);
  ASSERT_TRUE(events.front().birthFrameId.has_value());
  EXPECT_EQ(*events.front().birthFrameId, 10);
}

TEST(VioDiagnosticsLifecycle, EmitsOnlyRealInitialisationTransitions) {
  okvis::ViGraph graph;
  graph.setDiagnosticsCollectionEnabled(true);
  const okvis::LandmarkId landmarkId(2);
  ASSERT_TRUE(graph.addLandmark(
      landmarkId, Eigen::Vector4d(1.0, 0.0, 2.0, 1.0), false));
  graph.registerLandmarkBirth(
      landmarkId, okvis::diagnostics::EventContext{100, 1}, false);

  ASSERT_TRUE(graph.setLandmark(
      landmarkId, Eigen::Vector4d(1.0, 0.0, 2.0, 1.0), false));
  EXPECT_TRUE(graph.takeDiagnosticEvents().empty());

  ASSERT_TRUE(graph.setLandmark(
      landmarkId, Eigen::Vector4d(1.0, 0.0, 2.0, 1.0), true));
  auto events = graph.takeDiagnosticEvents();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().eventType,
            okvis::diagnostics::LandmarkEventType::Initialised);
  EXPECT_FALSE(events.front().initialisedBefore);
  EXPECT_TRUE(events.front().initialisedAfter);
}

TEST(VioDiagnosticsLifecycle, ClassifiesExplicitMergeAsLandmarkMerged) {
  okvis::ViGraph graph;
  graph.setDiagnosticsCollectionEnabled(true);
  const okvis::LandmarkId landmarkId(3);
  ASSERT_TRUE(graph.addLandmark(
      landmarkId, Eigen::Vector4d(0.0, 1.0, 2.0, 1.0), true));
  graph.registerLandmarkBirth(
      landmarkId, okvis::diagnostics::EventContext{100, 1}, false);

  ASSERT_TRUE(graph.removeLandmark(
      landmarkId,
      okvis::diagnostics::RemovalReason::ExplicitLandmarkMerge,
      okvis::diagnostics::EventContext{200, 2}));
  const auto events = graph.takeDiagnosticEvents();
  ASSERT_EQ(events.size(), 1);
  EXPECT_EQ(events.front().eventType,
            okvis::diagnostics::LandmarkEventType::LandmarkMerged);
  EXPECT_EQ(events.front().reason,
            okvis::diagnostics::RemovalReason::ExplicitLandmarkMerge);
}

}  // namespace
