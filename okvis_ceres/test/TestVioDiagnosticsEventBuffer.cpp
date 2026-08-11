#include <cstdint>

#include <gtest/gtest.h>

#include <okvis/ViGraph.hpp>
#include <okvis/ViSlamBackend.hpp>

namespace {

TEST(VioDiagnosticsEventBuffer, MovesLargeBatchInStrictInsertionOrder) {
  okvis::ViGraph graph;
  graph.setDiagnosticsGraphRole(okvis::diagnostics::GraphRole::Full);
  graph.setDiagnosticsCollectionEnabled(true);

  constexpr uint64_t kEventCount = 100000;
  for (uint64_t index = 1; index <= kEventCount; ++index) {
    graph.registerLandmarkBirth(
        okvis::LandmarkId(index),
        okvis::diagnostics::EventContext{index * 10, index}, true);
  }

  auto events = graph.takeDiagnosticEvents();
  ASSERT_EQ(events.size(), kEventCount);
  for (uint64_t index = 0; index < kEventCount; ++index) {
    EXPECT_EQ(events[index].landmarkId, index + 1);
    EXPECT_EQ(events[index].eventFrameId, index + 1);
    EXPECT_EQ(events[index].graphRole,
              okvis::diagnostics::GraphRole::Full);
  }
  EXPECT_TRUE(graph.takeDiagnosticEvents().empty());
}

TEST(VioDiagnosticsEventBuffer, BackendFlushIsSafeWhenDiagnosticsAreDisabled) {
  okvis::ViSlamBackend backend;
  EXPECT_NO_THROW(backend.flushDiagnostics());
}

}  // namespace
