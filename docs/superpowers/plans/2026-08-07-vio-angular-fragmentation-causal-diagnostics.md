# VIO Angular-Motion to Visual-Fragmentation Causal Diagnostics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 OKVIS2-X 增加默认关闭的结构化诊断输出，并建立事件对齐分析，直接检验“高角速度通过图像/匹配退化、弱三角化几何或预测误差进入 GP3P 失败与地图支撑崩溃”的中间链路。

**Architecture:** `okvis_common` 提供默认关闭、独占输出目录且无前后端类型依赖的 CSV writer；`okvis_frontend` 在多线程内部只做 thread-local 累计，join 后按 frame/source 写出；`okvis_ceres` 只把带原因和 graph role 的 landmark 生命周期事件写入 graph-local 内存队列，再由安全边界批量 drain。Python 工具验证 schema，按 OKVIS 实际图像时延对齐 IMU/mocap/轨迹，构造事件级中介量，并复用现有精度分析输出汇报图表。

**Tech Stack:** C++17, CMake, GoogleTest, Eigen, OpenCV, glog, Python 3, `unittest`, NumPy, SciPy, Matplotlib, existing OKVIS accuracy-analysis utilities.

---

## Execution Rules

- 用户已明确要求不得随意提交。执行本计划时不要运行 `git add`、`git commit`、`git reset` 或任何改写历史的命令；每个任务结束只检查 `git diff`。
- 保留 [Frontend.cpp](/home/chenguyuan/code/okvis_ws/src/OKVIS2-X/okvis_frontend/src/Frontend.cpp) 中现有未提交的 EUCM 改动，在其基础上增量修改。
- 不改变 RANSAC、匹配、三角化、landmark 清理或优化阈值。诊断发现现有行为可疑时先记录，不在本计划中顺手修复。
- 诊断默认关闭。未设置 `OKVIS_DIAGNOSTICS_DIR` 时不得创建文件，不得分配每特征诊断数组。
- 不在匹配 inner loop 中写文件或争用全局 mutex；只向 thread-local accumulator 写入，线程 join 后合并。
- `ViGraph` 的 landmark mutation 路径同样不得写文件；只追加到 graph-local 队列，在前端 frame flush 或 full-graph 批处理结束后批量写出。
- 所有 replay 使用新目录；runner 遇到非空目标目录必须失败，不允许自动删除或覆盖历史结果。
- `analyze_20260805.py` 已由用户确认过期并删除；不得恢复或依赖其 `.pyc`。仍在使用的 mocap 修正逻辑迁入独立公共模块。

## File Map

Create:

- `okvis_common/include/okvis/VioDiagnostics.hpp`: 稳定 schema、枚举、record 类型和 writer API。
- `okvis_common/src/VioDiagnostics.cpp`: 环境变量配置、CSV 输出、finite/empty 处理、metadata 完成标志。
- `okvis_common/test/TestVioDiagnostics.cpp`: writer 单元测试。
- `okvis_common/test/test_main.cpp`: `okvis_common_test` 入口。
- `okvis_frontend/include/okvis/FrontendDiagnostics.hpp`: 分位数、空间覆盖、frame/triangulation thread-local accumulator。
- `okvis_frontend/src/FrontendDiagnostics.cpp`: accumulator 实现。
- `okvis_frontend/test/TestFrontendDiagnostics.cpp`: accumulator 与覆盖率测试。
- `okvis_frontend/test/test_main.cpp`: `okvis_frontend_test` 入口。
- `okvis_ceres/test/TestVioDiagnosticsLifecycle.cpp`: lifecycle reason 与 graph role 测试。
- `okvis_ceres/test/TestVioDiagnosticsEventBuffer.cpp`: lifecycle 事件突发缓存和 drain 测试。
- `tools/accuracy_analysis/scripts/run_vio_diagnostics.py`: 安全、可恢复的 replay runner。
- `tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py`: runner 测试。
- `tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py`: schema 校验、事件对齐、中介统计和图表生成。
- `tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py`: 分析测试。
- `tools/accuracy_analysis/scripts/prepare_imu_time_offset_variants.py`: 不修改原始数据的离线时延干预数据视图。
- `tools/accuracy_analysis/scripts/test_prepare_imu_time_offset_variants.py`: 时延数据视图测试。
- `tools/accuracy_analysis/docs/VIO_CAUSAL_EXPERIMENT_PROTOCOL.md`: baseline、曝光、纹理和时延的正交实验协议。

Modify:

- `tools/accuracy_analysis/scripts/mocap_reference_correction.py`: 独立承载最终 BA 文件名、受影响序列和 mocap 刚体修正 helper。
- `okvis_common/CMakeLists.txt`: 编译 writer 并注册 `okvis_common_test`。
- `okvis_frontend/CMakeLists.txt`: 编译 accumulator 并注册 `okvis_frontend_test`。
- `okvis_frontend/include/okvis/Frontend.hpp`: 保存当前 frame accumulator；扩展匹配和 RANSAC 私有接口。
- `okvis_frontend/src/Frontend.cpp`: 检测、map matching、motion/spatial stereo、outlier、GP3P 与 frame flush 埋点。
- `okvis_ceres/include/okvis/ViGraph.hpp`: graph role 与带原因的内部移除接口。
- `okvis_ceres/include/okvis/ViGraphEstimator.hpp`: 传播 observation/landmark maintenance reason。
- `okvis_ceres/include/okvis/ViSlamBackend.hpp`: 前端可传入视觉拒绝原因。
- `okvis_ceres/src/ViGraph.cpp`: lifecycle event 的最终采集点。
- `okvis_ceres/src/ViGraphEstimator.cpp`: merge、marginalization 和 pose-graph conversion 原因传播。
- `okvis_ceres/src/ViSlamBackend.cpp`: realtime/full graph role 设置、原因分类和快照计数。
- `okvis_ceres/CMakeLists.txt`: 注册独立 diagnostics lifecycle/buffer 测试。
- `okvis_apps/src/okvis_app_synchronous.cpp`: 成功退出前 finalize diagnostics。
- `okvis_apps/src/okvis2x_app_synchronous.cpp`: 成功退出前 finalize diagnostics。
- `tools/accuracy_analysis/scripts/analyze_cross_sample_diagnostics.py`: 在新诊断结果完整时挂接中介证据摘要，不改变旧结果兼容性。
- `tools/accuracy_analysis/scripts/test_analyze_cross_sample_diagnostics.py`: 新挂接行为测试。
- `workspace/ego2_results/202608_week1_analysis/report.md`: 完成 replay 后更新证据，不在 C++ 实现阶段预写结论。

## Stable Diagnostic Contract

实现期间保持以下稳定名称：

```cpp
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

}  // namespace diagnostics
}  // namespace okvis
```

CSV filenames are fixed at schema version 1:

```text
vio_diag_metadata.csv
vio_diag_frame.csv
vio_diag_triangulation.csv
vio_diag_initialisation.csv
vio_diag_ransac.csv
vio_diag_landmark_events.csv
```

## Task 0: Replace the Obsolete Day-Specific Analysis Dependency

**Files:**

- Create: `tools/accuracy_analysis/scripts/mocap_reference_correction.py`
- Create: `tools/accuracy_analysis/scripts/test_mocap_reference_correction.py`
- Delete: `tools/accuracy_analysis/scripts/test_analyze_20260805.py`
- Modify: `tools/accuracy_analysis/scripts/analyze_multiday.py`
- Modify: `tools/accuracy_analysis/scripts/analyze_cross_sample_diagnostics.py`
- Modify: `tools/traj_visualization/plot_multiday_trajectories.py`

- [ ] **Step 1: Write failing shared-correction tests**

Create focused tests for the only day-specific behavior still consumed by
multiday tools:

```python
def test_session_fixed_lever_only_affects_confirmed_sequences():
    fixed = np.asarray([-0.1195, -0.0035, 0.1563])
    np.testing.assert_array_equal(
        correction.session_fixed_lever("20260805-122310", fixed), fixed
    )
    np.testing.assert_array_equal(
        correction.session_fixed_lever("20260805-114334", fixed), np.zeros(3)
    )

def test_correct_reference_positions_applies_body_fixed_lever():
    positions = np.zeros((2, 3))
    quaternions = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    corrected = correction.correct_reference_positions(
        positions, quaternions, np.asarray([1.0, 0.0, 0.0])
    )
    np.testing.assert_allclose(corrected, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
```

```bash
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_mocap_reference_correction -v
```

Expected: import failure because the shared module does not exist.

- [ ] **Step 2: Implement the focused shared module**

Move only `FINAL_BA_FILE`, `FIXED_DIAGNOSTIC_LEVER_M`, the three confirmed
affected sequence ids, `session_fixed_lever()`, `correct_reference_positions()`,
rigid alignment and `apply_effective_lever()` into the new module. Do not
restore obsolete day-specific plotting, alarm sweep or report generation.

- [ ] **Step 3: Migrate consumers and remove the obsolete test**

Import the new module as `day_analysis` in the three consumers to preserve their
existing internal references and mocks. Delete `test_analyze_20260805.py`; its
still-relevant lever cases now live in the focused shared-module test.

- [ ] **Step 4: Run the Python baseline tests**

```bash
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_mocap_reference_correction \
  tools.accuracy_analysis.scripts.test_analyze_multiday \
  tools.accuracy_analysis.scripts.test_analyze_cross_sample_diagnostics \
  tools.traj_visualization.test_plot_multiday_trajectories -v
```

Expected: all shared and consumer tests pass without importing
`analyze_20260805`.

- [ ] **Step 5: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis/scripts
git status --short
```

Expected: the recovered source is visible and no file is staged or committed.

## Task 1: Implement the Common Diagnostics Writer

**Files:**

- Create: `okvis_common/include/okvis/VioDiagnostics.hpp`
- Create: `okvis_common/src/VioDiagnostics.cpp`
- Create: `okvis_common/test/TestVioDiagnostics.cpp`
- Create: `okvis_common/test/test_main.cpp`
- Modify: `okvis_common/CMakeLists.txt`

- [ ] **Step 1: Add a failing disabled-mode test**

Create the test entry point using the repository's existing GoogleTest pattern:

```cpp
#include <glog/logging.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InitGoogleLogging(argv[0]);
  return RUN_ALL_TESTS();
}
```

Add a test that constructs a non-environment writer with an empty directory,
calls `configure(4)`, attempts every write method, calls `finish(true)`, and
asserts `enabled()==false` and no file exists in the test root.

- [ ] **Step 2: Register the failing test target**

Add `src/VioDiagnostics.cpp` to `okvis_common` and, under `BUILD_TESTS`, add:

```cmake
set(TEST_NAME okvis_common_test)
add_executable(${TEST_NAME}
  test/TestVioDiagnostics.cpp
  test/test_main.cpp
)
target_link_libraries(${TEST_NAME} ${LIB_NAME} ${GLOG_LIBRARIES} gtest)
target_compile_options(${TEST_NAME}
  PUBLIC ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE ${OKVIS_PRIVATE_CXX_FLAGS}
)
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
```

Run:

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_common_test -j2
```

Expected: compilation fails because `okvis/VioDiagnostics.hpp` and the writer
API do not exist.

- [ ] **Step 3: Define the complete public record API**

Define the enums in the Stable Diagnostic Contract and these records. Use
`std::optional<double>` for unavailable quantities; never encode unavailable
values as zero or NaN.

```cpp
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
  std::array<size_t, 10> observationsRemovedByReason{};
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
```

Expose this writer API:

```cpp
class VioDiagnostics {
 public:
  static constexpr int kSchemaVersion = 1;
  static VioDiagnostics& instance();

  explicit VioDiagnostics(std::string outputDirectory);
  ~VioDiagnostics();

  bool configure(size_t cameraCount,
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
```

`instance()` reads `OKVIS_DIAGNOSTICS_DIR`; an empty/unset value constructs a
disabled writer. `OKVIS_DIAGNOSTICS_OBSERVATION_ADDS=1` is the only value that
enables high-frequency add events. `writeLandmarkEvents()` assigns the global,
strictly increasing `event_sequence` while holding the writer lock once per
batch; mutation sites never call it directly.

- [ ] **Step 4: Implement deterministic CSV serialization**

Implement private overloads for `optional<double>` and `optional<uint64_t>`.
Both emit an empty field when missing; the floating-point overload also emits
empty for non-finite values. Use `std::locale::classic()` and
`std::setprecision(17)`. Generate per-camera columns at `configure()`:

```text
keypoints_cam0,response_p10_cam0,response_median_cam0,response_p90_cam0,
grid_fraction_cam0,hull_fraction_cam0,keypoints_cam1,response_p10_cam1
```

Append the matching funnel for every camera using the same suffix convention:

```text
projected_eligible_cam0,descriptor_comparisons_cam0,
descriptor_candidates_below_threshold_cam0,epipolar_rejected_cam0,
divergent_ray_rejected_cam0,accepted_initialised_cam0,
accepted_uninitialised_cam0,best_descriptor_distance_p10_cam0,
best_descriptor_distance_median_cam0,best_descriptor_distance_p90_cam0
```

For triangulation, emit one aggregate row per `source,camera0,camera1`; temporal
rows use the same camera index twice, spatial rows use a real pair, and
uninitialised-landmark rows use `camera1=-1` when observations span multiple
historical cameras.

At `configure()`, atomically create a `.vio_diagnostics.active` directory as an
exclusive ownership sentinel. Refuse configuration when the sentinel, a
`.vio_diagnostics.complete` sentinel, or any `vio_diag_*.csv` already exists;
never truncate an existing file. `finish(true)` flushes and closes all streams,
removes the active sentinel directory and exclusively creates an empty
`.vio_diagnostics.complete` file. On any
open/write failure, set `failed=true`, log one `LOG(ERROR)`, close all streams,
leave the active sentinel for recovery diagnosis, and make subsequent writes
no-ops. Do not throw into the estimator.

- [ ] **Step 5: Add writer contract tests**

Use
`std::filesystem::path(testing::TempDir()) / "vio_diag_writer_contract"`;
remove that test-owned child in fixture setup/teardown and assert:

```cpp
std::string readFirstLine(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::string line;
  std::getline(stream, line);
  return line;
}

std::vector<std::string> splitCsv(const std::string& line) {
  std::vector<std::string> fields;
  std::stringstream stream(line);
  std::string field;
  while (std::getline(stream, field, ',')) fields.push_back(field);
  return fields;
}

bool fileContains(const std::filesystem::path& path, const std::string& text) {
  std::ifstream stream(path);
  const std::string contents(
      std::istreambuf_iterator<char>(stream),
      std::istreambuf_iterator<char>());
  return contents.find(text) != std::string::npos;
}

const auto header = splitCsv(readFirstLine(root / "vio_diag_frame.csv"));
EXPECT_EQ(header.at(0), "schema_version");
EXPECT_EQ(header.at(1), "timestamp_ns");
EXPECT_EQ(header.at(2), "frame_id");
EXPECT_NE(std::find(header.begin(), header.end(), "keypoints_cam3"), header.end());
EXPECT_NE(std::find(header.begin(), header.end(),
                    "descriptor_comparisons_cam3"), header.end());
EXPECT_NE(std::find(header.begin(), header.end(),
                    "data_association_succeeded"), header.end());
EXPECT_TRUE(fileContains(root / "vio_diag_metadata.csv", "run_complete,true"));
EXPECT_TRUE(fileContains(root / "vio_diag_frame.csv", ",,"));
```

Cover idempotent `configure(4)`, rejection of `configure(3)` after headers were
created for four cameras, refusal of pre-existing CSV/active/complete outputs,
enum string stability, CSV escaping for metadata, empty output for infinity/NaN,
strict event-sequence assignment across two batches,
`finish(false)` not emitting `run_complete=true`, and write failure changing
`failed()` exactly once.

- [ ] **Step 6: Build and run the common tests**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_common_test -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_common_test$' --output-on-failure
```

Expected: build succeeds and `okvis_common_test` passes.

- [ ] **Step 7: Review without committing**

```bash
git diff --check -- okvis_common
git diff -- okvis_common/CMakeLists.txt okvis_common/include/okvis/VioDiagnostics.hpp okvis_common/src/VioDiagnostics.cpp
```

Expected: no whitespace errors; do not stage or commit.

## Task 2: Implement Frontend Accumulators

**Files:**

- Create: `okvis_frontend/include/okvis/FrontendDiagnostics.hpp`
- Create: `okvis_frontend/src/FrontendDiagnostics.cpp`
- Create: `okvis_frontend/test/TestFrontendDiagnostics.cpp`
- Create: `okvis_frontend/test/test_main.cpp`
- Modify: `okvis_frontend/CMakeLists.txt`

- [ ] **Step 1: Write failing distribution and merge tests**

Define tests before implementation:

```cpp
TEST(DiagnosticDistribution, UsesLinearInterpolatedQuantiles) {
  DiagnosticDistribution values;
  for (double value : {1.0, 2.0, 3.0, 4.0}) values.add(value);
  const auto summary = values.summary();
  EXPECT_DOUBLE_EQ(*summary.p10, 1.3);
  EXPECT_DOUBLE_EQ(*summary.median, 2.5);
  EXPECT_DOUBLE_EQ(*summary.p90, 3.7);
}

TEST(TriangulationAccumulator, MergePreservesCountsAndSamples) {
  TriangulationAccumulator a(TriangulationSource::SpatialStereo, 0, 1);
  TriangulationAccumulator b(TriangulationSource::SpatialStereo, 0, 1);
  a.recordAttempt(0.10, 0.02, 4.0, 3.0, true, false);
  b.recordAttempt(0.12, 0.03, 5.0, 4.0, true, true);
  a.merge(b);
  EXPECT_EQ(a.attempts(), 2);
  EXPECT_EQ(a.parallel(), 1);
  EXPECT_DOUBLE_EQ(*a.toRecord(10, 20).baselineM.median, 0.11);
}
```

- [ ] **Step 2: Define accumulator responsibilities**

Implement:

```cpp
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
  TriangulationAccumulator(
      TriangulationSource source, int camera0, int camera1);
  void recordAttempt(double baselineM,
                     double rayAngleRad,
                     double pixelDisplacementPx,
                     double depthM,
                     bool valid,
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
  TriangulationDiagnosticRecord toRecord(
      uint64_t timestampNs, uint64_t frameId) const;

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
           TriangulationAccumulator> triangulation;
  std::vector<InitialisationDiagnosticRecord> initialisation;
  std::vector<RansacDiagnosticRecord> ransac;
};

class FrontendDiagnosticFrames {
 public:
  explicit FrontendDiagnosticFrames(size_t cameraCount);
  void updateDetection(uint64_t timestampNs,
                       size_t cameraIndex,
                       const CameraDetectionAccumulator& detection);
  std::shared_ptr<FrontendFrameAccumulator> bindFrame(
      uint64_t frameId, uint64_t timestampNs);
  std::optional<FrontendFrameAccumulator> take(uint64_t frameId);
  void clear();

 private:
  size_t cameraCount_;
  std::mutex mutex_;
  std::map<uint64_t, FrontendFrameAccumulator> pendingByTimestamp_;
  std::map<uint64_t, std::shared_ptr<FrontendFrameAccumulator>> byFrameId_;
};
```

`DiagnosticDistribution::add()` ignores non-finite values. Quantiles sort a
copy and linearly interpolate at `(n-1)*q`, matching NumPy's default linear
method. Add merge tests proving a camera with zero accepted matches still
retains its projected, comparison, below-threshold and rejection counts; this
prevents accepted-match survivor bias from erasing total matching collapse.

- [ ] **Step 3: Implement keypoint spatial coverage**

Expose:

```cpp
CameraDetectionAccumulator summarizeKeypoints(
    const std::vector<cv::KeyPoint>& keypoints,
    int imageWidth,
    int imageHeight,
    int gridColumns = 4,
    int gridRows = 4);
```

Grid coverage is occupied cells divided by 16. Hull coverage is
`cv::contourArea(cv::convexHull(points))/(width*height)`, clamped to `[0,1]`;
fewer than three points yields zero. Add tests for empty points, one point,
four image corners, and out-of-bounds points clamped only for grid indexing.

- [ ] **Step 4: Add frontend test target and run it**

Register `okvis_frontend_test` in `okvis_frontend/CMakeLists.txt`, linking
`${LIB_NAME}`, `${GLOG_LIBRARIES}`, `gtest`, and `${OpenCV_LIBS}`.

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_frontend_test -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_frontend_test$' --output-on-failure
```

Expected: all accumulator and coverage tests pass.

- [ ] **Step 5: Review without committing**

```bash
git diff --check -- okvis_frontend/include/okvis/FrontendDiagnostics.hpp okvis_frontend/src/FrontendDiagnostics.cpp okvis_frontend/test okvis_frontend/CMakeLists.txt
```

Expected: no whitespace errors; do not stage or commit.

## Task 3: Instrument Detection, Map Matching, and Per-Frame State

**Files:**

- Modify: `okvis_frontend/include/okvis/Frontend.hpp`
- Modify: `okvis_frontend/src/Frontend.cpp`
- Modify: `okvis_frontend/test/TestFrontendDiagnostics.cpp`

- [ ] **Step 1: Add a failing frame-lifecycle test**

Test a small `FrontendDiagnosticFrames` owner that creates a frame once,
accepts camera updates in either order, and `take(frameId)` erases it:

```cpp
FrontendDiagnosticFrames frames(4);
CameraDetectionAccumulator cameraSummary;
cameraSummary.keypoints = 12;
frames.updateDetection(1000, 2, cameraSummary);
frames.updateDetection(1000, 0, cameraSummary);
frames.bindFrame(7, 1000);
auto completed = frames.take(7);
ASSERT_TRUE(completed.has_value());
EXPECT_EQ(completed->timestampNs, 1000);
EXPECT_EQ(completed->cameras.size(), 4);
EXPECT_FALSE(frames.take(7).has_value());
```

- [ ] **Step 2: Add diagnostics ownership to `Frontend`**

Add these private members without changing public behavior:

```cpp
std::unique_ptr<diagnostics::FrontendDiagnosticFrames> diagnosticFrames_;
```

In the constructor, call
`VioDiagnostics::instance().configure(numCameras_)`. Allocate
`diagnosticFrames_` only when the writer is enabled. In `clear()`, clear pending
accumulators without finalizing the process-wide writer.

- [ ] **Step 3: Record detection output after descriptor extraction**

At the end of `Frontend::detectAndDescribe()`, loop from zero to
`frameOut->numKeypoints(cameraIndex)`, retrieve each measurement with
`frameOut->getCvKeypoint(cameraIndex, keypointIndex, keypoint)`, call
`summarizeKeypoints()`, and update the frame accumulator using
`frameOut->timestamp().toNSec()` and `frameOut->id()`. Protect only the short
map update; do not hold the mutex while detecting or describing.

The archived EGO2 camera CSV contains timestamps and filenames but no exposure
duration. Write metadata `exposure_time_available=false`; downstream analysis
must call IMU integration between camera frames “frame-interval angular
exposure”, never “exposure-period angular displacement”. Image sharpness and
gradient proxies are computed offline from archived images in Task 9 so they
cannot perturb frontend timing.

Because a multiframe may not have its final backend state id during detection,
key the pending detection record by timestamp until `addStates()` assigns the
id, then bind timestamp to id at the start of
`dataAssociationAndInitialization()`. At that same entry point, before
`matchToMap()` or any visual optimisation, snapshot
`estimator.pose(StateId(framesInOut->id()))` into the immutable
`dataAssociationStartPose`. Name this source
`DataAssociationEntryAfterAddStates`: it is the estimator state entering visual
association, not a claim that no other sensor factor exists.

- [ ] **Step 4: Extend `matchToMap()` with thread-local counters**

Change the private worker signatures to accept one accumulator per matching
thread. Count:

```text
eligible landmarks
descriptor comparisons
descriptor candidates below the current threshold
epipolar-plane rejects
divergent-ray rejects
triangulation invalid/parallel results for uninitialised landmarks
accepted descriptor distance
predicted reprojection error before optimisation
accepted initialized and uninitialized map matches
```

Merge the thread accumulators after existing joins. Keep regular map matches
separate from calls with `loopClosureLandmarksToUseExclusively != nullptr`.
Populate every per-camera field in `FrameDiagnosticRecord`, including when the
accepted count is zero. Define `projectedEligible` after projection/FoV checks
and before descriptor pruning; define `descriptorComparisons` as actual BRISK
distance evaluations. Maintain a diagnostic-only per-keypoint best distance
initialized to infinity, independent of the production acceptance threshold;
after the worker loop, add every finite best value to
`bestMapDescriptorDistance` and count values below the unchanged BRISK threshold
as `descriptorCandidatesBelowThreshold` before epipolar/ray/uniqueness
rejection. Tests must cover a frame where comparisons and finite best distances
are nonzero while acceptance is zero.

- [ ] **Step 5: Classify non-RANSAC observation removals**

Pass explicit reasons at current frontend call sites:

```cpp
estimator.removeObservation(stateId, cameraIndex, keypointIndex,
    diagnostics::RemovalReason::LoopClosureReassociation);

estimator.removeObservation(stateId, cameraIndex, keypointIndex,
    diagnostics::RemovalReason::PostOptimisationReprojection);
```

Count rejection before the call so failed removals cannot silently disappear
from frame totals.

- [ ] **Step 6: Flush the frame after final cleanup**

Immediately after `estimator.cleanUnobservedLandmarks()`:

1. call `estimator.getLandmarks()`;
2. count active initialized/uninitialized landmarks from the realtime snapshot;
3. set tracking quality, keyframe, initialized,
   `tracking_quality_below_threshold = trackingQuality < 0.01`,
   `data_association_succeeded = trackingQuality >= 0.01`, and motion-stereo
   fields; do not serialize `trackingLost_`, which currently has no true-setting
   path;
4. write all triangulation, 2D-2D initialization and GP3P rows accumulated for
   that frame;
5. drain realtime graph lifecycle events through the backend batch API and
   write that batch once;
6. write exactly one frame row;
7. erase the accumulator.

If no detection accumulator exists, create an empty record and still write the
frame. This makes missing diagnostics observable instead of shifting rows.

- [ ] **Step 7: Build and run focused tests**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_frontend_test okvis_frontend -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^(okvis_common_test|okvis_frontend_test)$' --output-on-failure
```

Expected: tests pass and `Frontend.cpp` compiles with the existing EUCM branch
intact.

- [ ] **Step 8: Review without committing**

```bash
git diff --check -- okvis_frontend
git diff -- okvis_frontend/src/Frontend.cpp
```

Expected: EUCM changes remain present; no commit.

## Task 4: Instrument Temporal and Spatial Triangulation

**Files:**

- Modify: `okvis_frontend/src/Frontend.cpp`
- Modify: `okvis_frontend/include/okvis/Frontend.hpp`
- Modify: `okvis_frontend/test/TestFrontendDiagnostics.cpp`

- [ ] **Step 1: Add failing geometry-helper tests**

Add pure helper tests for physical quantities:

```cpp
EXPECT_NEAR(cameraBaseline(Eigen::Vector3d::Zero(),
                           Eigen::Vector3d(0.1, 0.0, 0.0)), 0.1, 1e-12);
EXPECT_NEAR(*rayAngle(Eigen::Vector3d::UnitX(),
                      Eigen::Vector3d::UnitY()),
            0.5 * std::acos(-1.0), 1e-12);
EXPECT_NEAR(pixelDisplacement(Eigen::Vector2d(1.0, 2.0),
                              Eigen::Vector2d(4.0, 6.0)), 5.0, 1e-12);
```

Expose and implement:

```cpp
double cameraBaseline(const Eigen::Vector3d& center0,
                      const Eigen::Vector3d& center1);
std::optional<double> rayAngle(const Eigen::Vector3d& ray0,
                               const Eigen::Vector3d& ray1);
double pixelDisplacement(const Eigen::Vector2d& point0,
                         const Eigen::Vector2d& point1);
```

Clamp the normalized ray dot product to `[-1,1]` before `acos`; return an empty
optional for a zero-norm ray rather than recording a fabricated angle.

- [ ] **Step 2: Instrument temporal motion stereo**

In `matchMotionStereo()` create one thread-local accumulator per worker and
camera. For every descriptor candidate reaching triangulation, record:

```cpp
const double baseline = (T_WC1.r() - T_WC0.r()).norm();
const double angle = diagnostics::rayAngle(e0_W, e1_W);
const double displacement = (pt1 - pt0).norm();
```

Record `isValid`, `isParallel`, finite depth, descriptor threshold and every
post-triangulation rejection category. Merge only after all workers join.
Record births and false-to-true initialization transitions at the points where
`addLandmark()` and
`setLandmark(landmarkId, homogeneousPoint, true)` actually succeed.

- [ ] **Step 3: Instrument same-frame spatial stereo**

In `matchStereo()` aggregate by `(SpatialStereo, im0, im1)`. Use actual
`T_WC0.r()/T_WC1.r()` and bearing rays. Pixel displacement between different
cameras is recorded only as a descriptive image-coordinate displacement; it is
not called temporal optical flow in Python or the report.

Count merge, birth, observation-add, initialization transition, projection
failure and `<4 px` rejection separately.

- [ ] **Step 4: Instrument uninitialized-landmark re-triangulation**

At the `triangulateFast()` call in `matchToMapByThreadUnitialised()`, aggregate
under `UninitialisedLandmark`. Preserve historical camera/frame identities in
the internal accumulator long enough to compute actual camera-center baseline
and ray angle; emit `camera1=-1` only when one output row combines several
historical cameras.

- [ ] **Step 5: Assert accounting invariants**

Add debug-only assertions and unit tests for:

```text
attempts >= valid + invalid
initialisable <= valid
parallel <= attempts
landmark_initialisations <= initialisable
```

Do not require `valid+invalid==attempts` because back-projection and early
direction checks may reject a candidate before `triangulateFast()`.

- [ ] **Step 6: Build and test**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_frontend_test okvis_frontend -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_frontend_test$' --output-on-failure
```

Expected: tests pass; no diagnostic writes occur in worker loops.

- [ ] **Step 7: Review without committing**

```bash
rg -n "writeTriangulation|writeFrame|ofstream|fstream" okvis_frontend/src/Frontend.cpp
git diff --check -- okvis_frontend
```

Expected: writer calls appear only at the post-join/frame-flush layer; no commit.

## Task 5: Instrument 2D-2D Initialisation and GP3P Prediction Consistency

**Files:**

- Modify: `okvis_frontend/include/okvis/Frontend.hpp`
- Modify: `okvis_frontend/src/Frontend.cpp`
- Modify: `okvis_frontend/test/TestFrontendDiagnostics.cpp`

- [ ] **Step 1: Add failing 2D-2D model-selection tests**

Add a pure classifier that mirrors the existing `runRansac2d2d()` branch
without treating its fallback `rotationOnly = true` as a successful
rotation-only estimate:

```cpp
const auto rotation = classifyInitialisationModels(30, true, 27, true, 18);
EXPECT_EQ(rotation.selection, InitialisationModelSelection::RotationOnly);
EXPECT_TRUE(rotation.successful);

const auto relative = classifyInitialisationModels(30, true, 15, true, 27);
EXPECT_EQ(relative.selection, InitialisationModelSelection::RelativePose);
EXPECT_TRUE(relative.successful);

const auto failed = classifyInitialisationModels(30, false, 0, false, 0);
EXPECT_EQ(failed.selection, InitialisationModelSelection::None);
EXPECT_FALSE(failed.successful);
```

Preserve the current thresholds: rotation-only is selected when its ratio is
higher than relative-pose or above `0.8`; rotation-only success requires more
than 10 inliers, while relative-pose success requires more than 10 inliers and
ratio above `0.8`.

- [ ] **Step 2: Record every 2D-2D camera comparison**

In `runRansac2d2d()`, append one `InitialisationDiagnosticRecord` for each
`(currentFrameId, olderFrameId, camera)`, including `<10 correspondence`
skips. Store both `computeModel()` results, both inlier counts/ratios, the
selected branch, selected inliers, and the function's final return result.
Build records locally and append only after the camera loop, so the final
function result is consistent on every row.

Pass `Initialisation2d2dOutlier` for removed observations. Keep these rows in
`vio_diag_initialisation.csv`; do not mix them with runtime 3D-2D GP3P rows.
Add a test proving complete model failure serializes `selection=none` even
though current production code ultimately sets its output `rotationOnly=true`.

- [ ] **Step 3: State the E-matrix diagnostic limit explicitly**

The OpenGV `STEWENIUS` branch estimates a constrained relative-pose model. Do
not claim that its returned essential matrix norm directly measures
observability or that pure rotation must serialize `E=0`; normalization and
model constraints can hide that algebraic symptom. Test H2 using the joint
evidence of rotation-only versus relative-pose support, accepted temporal ray
angles/baselines, initialisable landmark rate, and external local
translation/rotation. The report must state that this 2D-2D path is used during
initialisation, while the runtime `RANSAC FAIL` log is 3D-2D GP3P.

- [ ] **Step 4: Add a failing GP3P outcome-semantics test**

Introduce a pure result classifier and test the current early-return behavior
without changing it:

```cpp
const auto noPrior = classifyRansacOutcome(false, 0, 0, false);
EXPECT_EQ(noPrior.status, RansacStatus::NoPriorFrame);
EXPECT_FALSE(noPrior.returnedSuccess);

const auto insufficient = classifyRansacOutcome(true, 7, 0, false);
EXPECT_EQ(insufficient.status, RansacStatus::InsufficientCorrespondences);
EXPECT_FALSE(insufficient.thresholdSuccess);
EXPECT_TRUE(insufficient.returnedSuccess);  // preserves current bool conversion

const auto rejected = classifyRansacOutcome(true, 20, 9, true);
EXPECT_EQ(rejected.status, RansacStatus::ThresholdRejected);
EXPECT_FALSE(rejected.returnedSuccess);
```

Define the helper contract in `FrontendDiagnostics.hpp`:

```cpp
struct RansacOutcome {
  RansacStatus status;
  bool thresholdSuccess;
  bool returnedSuccess;
};

RansacOutcome classifyRansacOutcome(
    bool hasPriorFrame,
    size_t correspondences,
    size_t inliers,
    bool modelComputed);
```

For fewer than 10 correspondences, preserve the current conversion
`returnedSuccess = static_cast<bool>(correspondences)`. Otherwise success
requires `modelComputed`, at least 10 inliers and inlier ratio greater than
0.7.

- [ ] **Step 5: Pass primary and compound triggers into `runRansac3d2d()`**

Change the private signature to:

```cpp
bool runRansac3d2d(
    Estimator& estimator,
    const cameras::NCameraSystem& nCameraSystem,
    std::shared_ptr<MultiFrame> currentFrame,
    bool initializePose,
    bool removeOutliers,
    diagnostics::RansacTrigger primaryTrigger,
    uint32_t triggerMask,
    const kinematics::Transformation& dataAssociationStartPose);
```

At call sites select `NoImu`, `LargeReprojectionError`,
`TooFewAcceptedMatches`, or `RetryWithUninitialisedLandmarks` from the actual
branch that caused the call. Set one bit for every simultaneously true
condition, so “large reprojection error plus too few accepted matches” is not
collapsed to one cause. The primary trigger describes the branch that issued
the call; do not infer it afterward from outcome.

- [ ] **Step 6: Record every GP3P invocation and all pose snapshots**

Before both early returns (`numFrames()<2` and `<10` correspondences), create a
record with status, correspondence count, per-camera counts, occupied-grid
fraction, primary trigger, trigger mask, `modelComputed=false`,
`thresholdSuccess=false`, and the exact bool that the current function returns.
Set `invocation` to the current frame accumulator's GP3P row count before
appending the new record, so numbering restarts at zero for every frame.

After `computeModel(0)`, record its boolean result, inlier list, per-camera
inliers and all threshold decisions. For each GP3P-removed observation pass
`RemovalReason::Gp3pOutlier` to the backend.

- [ ] **Step 7: Record both frame-entry and immediate GP3P corrections**

Copy the immutable frame-entry pose into `dataAssociationStartPose`, snapshot
the estimator pose immediately before each invocation into `preInvocationPose`,
and store the finite GP3P model as `gp3pModelPose`. Compute both deltas:

```cpp
const auto T_start_model = T_WS_data_association_start.inverse() * T_WS_gp3p;
const auto T_pre_model = T_WS_pre_invocation.inverse() * T_WS_gp3p;
record.startToModelTranslationM = T_start_model.r().norm();
record.startToModelRotationRad = Eigen::AngleAxisd(T_start_model.C()).angle();
record.preInvocationToModelTranslationM = T_pre_model.r().norm();
record.preInvocationToModelRotationRad =
    Eigen::AngleAxisd(T_pre_model.C()).angle();
```

Do this even when thresholds reject the model. The first delta tests consistency
with the state entering visual association; the second isolates the immediate
GP3P innovation. The second invocation occurs after visual optimisation and
must never be labelled “raw IMU propagated pose”. Do not call `setPose()` unless
the original success branch would have called it.

- [ ] **Step 8: Build and test**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_frontend_test okvis_frontend -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_frontend_test$' --output-on-failure
```

Expected: model-selection and outcome tests pass; compiler confirms all call
sites provide pose and trigger context.

- [ ] **Step 9: Review without committing**

```bash
rg -n "runRansac(2d2d|3d2d)\(" okvis_frontend/include/okvis/Frontend.hpp okvis_frontend/src/Frontend.cpp
git diff --check -- okvis_frontend
```

Expected: both paths have one declaration, one definition and only the known
call sites; no commit.

## Task 6: Add Reasoned Landmark Lifecycle Events

**Files:**

- Modify: `okvis_ceres/include/okvis/ViGraph.hpp`
- Modify: `okvis_ceres/include/okvis/ViGraphEstimator.hpp`
- Modify: `okvis_ceres/include/okvis/ViSlamBackend.hpp`
- Modify: `okvis_ceres/src/ViGraph.cpp`
- Modify: `okvis_ceres/src/ViGraphEstimator.cpp`
- Modify: `okvis_ceres/src/ViSlamBackend.cpp`
- Create: `okvis_ceres/test/TestVioDiagnosticsLifecycle.cpp`
- Create: `okvis_ceres/test/TestVioDiagnosticsEventBuffer.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Add failing removal-reason tests**

Build a graph with one landmark and two observations, advance its newest state
from timestamp 100 to 500, remove the old observation with
`PostOptimisationReprojection`, call `takeDiagnosticEvents()`, and assert the
buffered event contains:

```text
graph_role=realtime
event_type=observation_removed
reason=post_optimisation_reprojection
event_timestamp_ns=500
event_frame_id=<newest processing state>
subject_timestamp_ns=100
subject_frame_id=<removed observation state>
birth_timestamp_ns=100
observations_before=2
observations_after=1
```

Add a second test proving `StateMarginalisation` rows are not mislabeled as
visual rejection. Assert event order from the moved buffer, not subject-state
time. The graph unit test must not configure or touch the process CSV writer.

- [ ] **Step 2: Give each graph a stable role**

Add to `ViGraph`:

```cpp
void setDiagnosticsGraphRole(diagnostics::GraphRole role) {
  diagnosticsGraphRole_ = role;
}

void setDiagnosticsCollectionEnabled(bool enabled);

diagnostics::GraphRole diagnosticsGraphRole_ =
    diagnostics::GraphRole::Realtime;
```

Also add graph-local storage:

```cpp
struct LandmarkBirthContext {
  uint64_t timestampNs = 0;
  uint64_t frameId = 0;
};

struct DiagnosticsState {
  std::vector<diagnostics::LandmarkEventRecord> events;
  std::map<LandmarkId, LandmarkBirthContext> births;
};

std::unique_ptr<DiagnosticsState> diagnostics_;

std::vector<diagnostics::LandmarkEventRecord> takeDiagnosticEvents() {
  if (!diagnostics_) return {};
  std::vector<diagnostics::LandmarkEventRecord> events;
  events.swap(diagnostics_->events);
  return events;
}
```

These containers exist only when diagnostics are enabled. `ViGraph` mutation
methods append records to this vector under the graph's existing ownership;
they never call `VioDiagnostics` and never perform file I/O.

In the `ViSlamBackend` constructor, set `realtimeGraph_` to `Realtime` and
`fullGraph_` to `Full` before states or landmarks are added, then enable both
buffers from the already configured writer's `enabled()` value. Pure graph
tests call `setDiagnosticsCollectionEnabled(true)` directly and therefore do
not instantiate the process writer.

- [ ] **Step 3: Propagate explicit maintenance reasons**

Change internal mutation signatures to require reason and event context. Keep
source compatibility only in public wrappers, where the wrapper supplies
`Unknown` and derives current context when an old caller omits a reason:

```cpp
bool removeObservation(
    KeypointIdentifier keypointId,
    diagnostics::RemovalReason reason,
    const diagnostics::EventContext& context);
bool removeLandmark(
    LandmarkId landmarkId,
    diagnostics::RemovalReason reason,
    const diagnostics::EventContext& context);
int cleanUnobservedLandmarks(
    std::map<LandmarkId, std::set<KeypointIdentifier>>* removed,
    diagnostics::RemovalReason reason,
    const diagnostics::EventContext& context);
bool removeAllObservations(
    StateId stateId,
    diagnostics::RemovalReason reason,
    const diagnostics::EventContext& context);
bool mergeLandmark(
    LandmarkId fromId,
    LandmarkId intoId,
    std::map<StateId, MultiFramePtr>& multiFrames,
    diagnostics::RemovalReason reason,
    const diagnostics::EventContext& context);
```

`EventContext` contains the newest processing frame id/timestamp at the time of
mutation. The graph derives `subjectFrameId/subjectTimestampNs` from the
observation or landmark being changed and looks up the landmark's recorded
birth context. Compatibility wrappers may construct the event context from the
newest graph state, but production call sites must pass explicit reasons.

Extend the public backend wrapper used by the frontend:

```cpp
bool removeObservation(
    StateId stateId,
    size_t cameraIndex,
    size_t keypointIndex,
    diagnostics::RemovalReason reason =
        diagnostics::RemovalReason::Unknown);
```

The wrapper forwards the same reason to full and realtime graphs, while the
graph's stable role keeps their rows distinguishable.

At every call site pass one of the stable reasons. Use
`StateMarginalisation` for IMU-frame elimination,
`PoseGraphConversion` for conversion/removal of old keyframes,
`RealtimeFullGraphSync` only for copying realtime state into full graph, and
`ExplicitLandmarkMerge` for both single and vector merge paths.

- [ ] **Step 4: Emit events at mutation points**

In `ViGraph::removeObservation()`, capture landmark id, initialized flag,
quality and observation counts before mutation, then enqueue one event after
successful mutation. Use current `EventContext` for `event_*`, the removed
observation state for `subject_*`, and the landmark birth map for `birth_*`.
In `removeLandmark()` and `cleanUnobservedLandmarks()`, enqueue a landmark
removal event even for a zero-observation landmark.

Emit both graph roles, but preserve `graph_role` so analysis can use realtime
as the primary stream and treat full-graph rows as synchronization audit only.

- [ ] **Step 5: Record births and initialization transitions**

In `ViSlamBackend::addLandmark()`, register the same birth context in both
graphs after both updates succeed and enqueue one realtime `Birth` event; do
not enqueue a duplicate full-graph birth for the same logical operation. In
`setLandmark()` compare initialized state before/after and enqueue
`Initialised` only on `false -> true`.

When `ViGraph::updateLandmarks()` changes initialized state after optimization,
enqueue the transition using the newest state id/timestamp. Emit `Initialised` for
`false -> true` and `Deinitialised` for `true -> false`; never fold the latter
into the initialization-success count.

- [ ] **Step 6: Gate high-frequency observation-add events**

In the existing templated `ViSlamBackend::addObservation()` wrapper, emit
`ObservationAdded` only when
`VioDiagnostics::instance().observationAddsEnabled()` is true. At each frontend
call site, increment the current frame's add counter only after
`addObservation()` succeeds, so population replay remains complete with
detailed add events disabled.

- [ ] **Step 7: Batch-drain events only at safe boundaries**

Expose a backend method that moves, rather than copies, queued rows:

```cpp
std::vector<diagnostics::LandmarkEventRecord>
takeLandmarkDiagnosticEvents(diagnostics::GraphRole role);
```

Drain realtime events once during the frontend frame flush. Drain full-graph
events after its optimisation/marginalisation batch releases the graph's hot
mutation path and once more during application finalization. Pass each moved
batch to `VioDiagnostics::writeLandmarkEvents()`. Never hold a graph lock while
performing CSV I/O.

Add a stress test that enqueues and drains 100,000 synthetic removal events,
asserts exact order/count and an empty second drain, and confirms mutation does
not instantiate the writer or create files.

- [ ] **Step 8: Build and run Ceres tests**

Register a separate `okvis_ceres_diagnostics_test` executable containing
`test/TestVioDiagnosticsLifecycle.cpp` and `test/test_main.cpp`, linked with
`${LIB_NAME}`, `${GLOG_LIBRARIES}` and `gtest`. Do not add it to the existing
`okvis_ceres_test`: a separate process isolates the optional writer batch-drain
integration fixture, while pure graph-buffer tests remain independent of the
process singleton.

```cmake
set(DIAGNOSTICS_TEST_NAME okvis_ceres_diagnostics_test)
add_executable(${DIAGNOSTICS_TEST_NAME}
  test/TestVioDiagnosticsLifecycle.cpp
  test/TestVioDiagnosticsEventBuffer.cpp
  test/test_main.cpp
)
target_link_libraries(${DIAGNOSTICS_TEST_NAME}
  ${LIB_NAME}
  ${GLOG_LIBRARIES}
  gtest
)
target_compile_options(${DIAGNOSTICS_TEST_NAME}
  PUBLIC ${OKVIS_PUBLIC_CXX_FLAGS}
  PRIVATE ${OKVIS_PRIVATE_CXX_FLAGS}
)
add_test(NAME ${DIAGNOSTICS_TEST_NAME} COMMAND ${DIAGNOSTICS_TEST_NAME})
```

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_ceres_test okvis_ceres_diagnostics_test -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_ceres(_diagnostics)?_test$' --output-on-failure
```

Expected: lifecycle tests and existing Ceres tests pass.

- [ ] **Step 9: Audit every removal call and review without committing**

```bash
rg -n "removeObservation\(|removeLandmark\(|removeAllObservations\(|cleanUnobservedLandmarks\(" okvis_frontend okvis_ceres -g '*.cpp' -g '*.hpp'
git diff --check -- okvis_ceres okvis_frontend
```

Expected: frontend visual removals and backend maintenance paths have explicit
reasons; only compatibility/test-only paths may use `Unknown`; no commit.

## Task 7: Finalize Diagnostics on Successful Application Exit

**Files:**

- Modify: `okvis_apps/src/okvis_app_synchronous.cpp`
- Modify: `okvis_apps/src/okvis2x_app_synchronous.cpp`
- Modify: `okvis_common/test/TestVioDiagnostics.cpp`

- [ ] **Step 1: Add finish idempotency tests**

Call `finish(true)` twice and assert there is exactly one
`run_complete,true` row. Destroying an unfinished writer must flush rows but
must not add `run_complete=true`.

- [ ] **Step 2: Add explicit successful finalization**

After final trajectory/map writes have succeeded and immediately before
`return EXIT_SUCCESS`, add:

```cpp
okvis::diagnostics::VioDiagnostics::instance().finish(true);
```

Do not call `finish(true)` on usage, dataset-reader, vocabulary, or other error
returns. Writer failure remains non-fatal to VIO, but metadata will lack a
successful completion marker and analysis will reject the run.

- [ ] **Step 3: Record relevant static metadata**

Once VI parameters are loaded, write camera count, matching threshold,
RANSAC minimum inliers/ratio, triangulation parallel rule version, executable
name, and optional runner-provided `OKVIS_DIAGNOSTICS_RUN_ID` and
`OKVIS_DIAGNOSTICS_BUILD_ID`.

- [ ] **Step 4: Build both synchronous apps**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target okvis_app_synchronous okvis2x_app_synchronous -j2
```

Expected: both apps link with `okvis_common`; no warnings introduced by the new code.

- [ ] **Step 5: Review without committing**

```bash
git diff --check -- okvis_apps okvis_common
```

Expected: no whitespace errors; no commit.

## Task 8: Implement the Safe Replay Runner

**Files:**

- Create: `tools/accuracy_analysis/scripts/run_vio_diagnostics.py`
- Create: `tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py`

- [ ] **Step 1: Write failing CLI and safety tests**

Use `unittest.mock.patch("subprocess.run")` to verify the exact command and
environment. Cover refusal of a nonempty output directory, missing dataset,
missing binary/config, missing or duplicate mocap reference logs, nonzero
process exit, absent `run_complete=true`, and a successful dry-run command.

Expected command:

```python
[
    str(binary),
    str(config),
    str(dataset),
    str(run_dir),
]
```

Expected environment additions:

```python
{
    "OKVIS_DIAGNOSTICS_DIR": str(run_dir / "diagnostics"),
    "OKVIS_DIAGNOSTICS_RUN_ID": f"{sequence}-{run_name}",
    "OKVIS_DIAGNOSTICS_BUILD_ID": build_id,
    "QT_QPA_PLATFORM": "offscreen",
}
```

- [ ] **Step 2: Implement a deterministic CLI**

Use:

```text
--binary /home/chenguyuan/code/okvis_ws/build/okvis/okvis_app_synchronous
--config config/okvis2_eucm_EGO2.yaml
--data-root /home/chenguyuan/data
--reference-results-root workspace/ego2_results
--results-root workspace/ego2_results/202608_causal_diagnostics
--sequences 20260806-175103 20260806-175304 20260806-175539
--repeats 2
--jobs 1
--dry-run
```

Map a sequence prefix `YYYYMMDD` to
`<data-root>/<day>/<sequence>_euroc`. Create a fresh
`<results-root>/<sequence>/runN` only after all inputs validate. Capture merged
stdout/stderr to `run.log` without shell invocation.

Resolve mocap from
`<reference-results-root>/<day>/**/<sequence>/mocap_<sequence>.log` and require
exactly one match. Persist its resolved absolute path and the resolved config
path in every run manifest; neither the EuRoC dataset nor diagnostics directory
is assumed to contain mocap.

Use `git rev-parse --verify HEAD` plus a dirty suffix from
`git status --porcelain` for metadata only; do not stage or commit.

- [ ] **Step 3: Validate outputs after each process**

Require process return code zero, the six diagnostic CSVs, both online and
final-BA trajectories, and metadata `run_complete=true`. Write a runner-owned
`run_manifest.json` atomically with command, inputs, build id, start/end time,
return code, `mocap_path`, `config_path`, parsed `image_delay_s`, and produced
files. Also require `.vio_diagnostics.complete` and reject a leftover
`.vio_diagnostics.active` sentinel.

Never infer success from the existence of a directory alone.

- [ ] **Step 4: Run unit tests and dry-run**

```bash
python3 -m unittest tools.accuracy_analysis.scripts.test_run_vio_diagnostics -v
python3 tools/accuracy_analysis/scripts/run_vio_diagnostics.py --dry-run --sequences 20260806-175103 20260806-175304 20260806-175539 --repeats 2
```

Expected: tests pass; dry-run prints six commands and creates no run directories.

- [ ] **Step 5: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis/scripts/run_vio_diagnostics.py tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py
```

Expected: no whitespace errors; no commit.

## Task 9: Implement Schema Validation and Time Alignment

**Files:**

- Create: `tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py`
- Create: `tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/mocap_reference_correction.py`
- Modify: `tools/accuracy_analysis/scripts/test_mocap_reference_correction.py`

- [ ] **Step 1: Write failing schema-validation tests**

Build minimal CSV fixtures and test rejection of missing files, wrong schema,
duplicate `(frame_id,source,camera0,camera1)` rows, non-monotonic frame
timestamps, non-increasing event sequences, missing `run_complete=true`, writer
failure metadata, missing manifest mocap/config paths, config image-delay
mismatch, and camera column count mismatch. Parameterize the GP3P fixture to
drop each pose component, optional model-pose column, correction column and
model-status column in turn; every incomplete header must be rejected even
when all row values for an optional model are empty.

- [ ] **Step 2: Implement typed loaders**

Import `csv`, `dataclasses`, `json`, `math`, `Path`, `numpy as np`, and
`PIL.Image`, then define:

```python
@dataclasses.dataclass(frozen=True)
class DiagnosticRun:
    sequence: str
    run: str
    root: Path
    manifest: dict[str, object]
    metadata: dict[str, str]
    frames: list[dict[str, object]]
    triangulation: list[dict[str, object]]
    initialisation: list[dict[str, object]]
    ransac: list[dict[str, object]]
    landmark_events: list[dict[str, object]]


REQUIRED_COLUMNS = {
    "vio_diag_frame.csv": {
        "schema_version", "timestamp_ns", "frame_id", "initialised",
        "data_association_succeeded", "tracking_quality_below_threshold",
        "keyframe", "tracking_quality", "projected_eligible_cam0",
        "descriptor_comparisons_cam0",
        "descriptor_candidates_below_threshold_cam0",
        "accepted_initialised_cam0", "accepted_uninitialised_cam0",
        "active_initialised_landmarks", "active_uninitialised_landmarks",
    },
    "vio_diag_triangulation.csv": {
        "schema_version", "timestamp_ns", "frame_id", "source", "camera0",
        "camera1", "attempts", "valid", "invalid", "parallel",
    },
    "vio_diag_initialisation.csv": {
        "schema_version", "timestamp_ns", "current_frame_id",
        "older_frame_id", "camera", "invocation", "correspondences",
        "rotation_model_computed", "rotation_inliers",
        "rotation_inlier_ratio", "relative_pose_model_computed",
        "relative_pose_inliers", "relative_pose_inlier_ratio",
        "selected_model", "selected_model_successful",
        "selected_inliers", "function_returned_success",
        "function_return_value",
    },
    "vio_diag_ransac.csv": {
        "schema_version", "timestamp_ns", "frame_id", "invocation",
        "primary_trigger", "trigger_mask", "status", "correspondences",
        "inliers", "inlier_ratio", "returned_success",
        "model_computed", "threshold_success",
        "data_association_start_pose_source", "pre_invocation_pose_source",
        "data_association_start_tx", "data_association_start_ty",
        "data_association_start_tz", "data_association_start_qw",
        "data_association_start_qx", "data_association_start_qy",
        "data_association_start_qz", "pre_invocation_tx",
        "pre_invocation_ty", "pre_invocation_tz", "pre_invocation_qw",
        "pre_invocation_qx", "pre_invocation_qy", "pre_invocation_qz",
        "gp3p_model_tx", "gp3p_model_ty", "gp3p_model_tz",
        "gp3p_model_qw", "gp3p_model_qx", "gp3p_model_qy",
        "gp3p_model_qz", "start_to_model_rotation_rad",
        "start_to_model_translation_m",
        "pre_invocation_to_model_rotation_rad",
        "pre_invocation_to_model_translation_m",
    },
    "vio_diag_landmark_events.csv": {
        "schema_version", "event_sequence", "event_timestamp_ns",
        "event_frame_id", "subject_timestamp_ns", "subject_frame_id",
        "birth_timestamp_ns", "birth_frame_id", "landmark_id", "graph_role",
        "event_type", "reason", "observations_before", "observations_after",
    },
}


def _read_rows(path: Path, required: set[str]) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"missing diagnostic file: {path}")
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        missing = required - fields
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        return list(reader)


def _convert_scalar(key: str, value: str) -> object:
    if value == "":
        return None
    if key.endswith("_ns") or key in {
        "schema_version", "frame_id", "landmark_id", "invocation",
        "event_sequence", "current_frame_id", "older_frame_id", "camera",
        "event_frame_id", "subject_frame_id", "birth_frame_id", "trigger_mask",
        "camera0", "camera1", "correspondences", "inliers", "attempts",
        "rotation_inliers", "relative_pose_inliers", "selected_inliers",
        "function_return_value", "outliers", "removed_observations",
        "valid", "invalid", "parallel", "observations_before",
        "observations_after", "active_initialised_landmarks",
        "active_uninitialised_landmarks",
    } or key.startswith((
        "keypoints_cam", "projected_eligible_cam",
        "descriptor_comparisons_cam",
        "descriptor_candidates_below_threshold_cam",
        "epipolar_rejected_cam", "divergent_ray_rejected_cam",
        "accepted_initialised_cam", "accepted_uninitialised_cam",
    )):
        return int(value)
    if key in {
        "initialised", "data_association_succeeded",
        "tracking_quality_below_threshold", "keyframe", "returned_success",
        "rotation_model_computed", "relative_pose_model_computed",
        "selected_model_successful", "function_returned_success",
        "model_computed", "threshold_success",
    }:
        if value not in {"0", "1"}:
            raise ValueError(f"{key}: expected 0 or 1, got {value!r}")
        return value == "1"
    try:
        converted = float(value)
    except ValueError:
        return value
    if not math.isfinite(converted):
        raise ValueError(f"{key}: non-finite numeric value {value!r}")
    return converted


def _typed_rows(path: Path, required: set[str]) -> list[dict[str, object]]:
    return [
        {key: _convert_scalar(key, value) for key, value in row.items()}
        for row in _read_rows(path, required)
    ]


def load_diagnostic_run(root: Path, sequence: str, run: str) -> DiagnosticRun:
    run_root = root / sequence / run
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"missing run manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("mocap_path", "config_path", "image_delay_s"):
        if key not in manifest:
            raise ValueError(f"{manifest_path}: missing {key}")
    for key in ("mocap_path", "config_path"):
        if not Path(str(manifest[key])).is_file():
            raise ValueError(f"{manifest_path}: {key} does not exist")
    image_delay_s = float(manifest["image_delay_s"])
    if not math.isfinite(image_delay_s):
        raise ValueError(f"{manifest_path}: invalid image_delay_s")
    diagnostics = run_root / "diagnostics"
    if not (diagnostics / ".vio_diagnostics.complete").is_file():
        raise ValueError(f"{diagnostics}: missing completion sentinel")
    if (diagnostics / ".vio_diagnostics.active").exists():
        raise ValueError(f"{diagnostics}: active writer sentinel remains")
    metadata_rows = _read_rows(
        diagnostics / "vio_diag_metadata.csv",
        {"schema_version", "key", "value"},
    )
    metadata = {row["key"]: row["value"] for row in metadata_rows}
    if metadata.get("run_complete") != "true":
        raise ValueError(f"{diagnostics}: run is not complete")
    if metadata.get("writer_failed", "false") == "true":
        raise ValueError(f"{diagnostics}: writer reported failure")
    tables = {
        name: _typed_rows(diagnostics / name, required)
        for name, required in REQUIRED_COLUMNS.items()
    }
    for name, rows in tables.items():
        versions = {row["schema_version"] for row in rows}
        if versions and versions != {1}:
            raise ValueError(f"{name}: unsupported schema versions {versions}")
    return DiagnosticRun(
        sequence=sequence,
        run=run,
        root=run_root,
        manifest=manifest,
        metadata=metadata,
        frames=tables["vio_diag_frame.csv"],
        triangulation=tables["vio_diag_triangulation.csv"],
        initialisation=tables["vio_diag_initialisation.csv"],
        ransac=tables["vio_diag_ransac.csv"],
        landmark_events=tables["vio_diag_landmark_events.csv"],
    )
```

After loading, apply explicit uniqueness and monotonicity helpers: frame keys
are unique by `frame_id`; triangulation keys by
`(frame_id,source,camera0,camera1)`; initialisation keys by
`(current_frame_id,older_frame_id,camera,invocation)`; GP3P keys by
`(frame_id,invocation)`. Event rows may share frame and landmark, but
`event_sequence` must be strictly increasing in file order. Do not require
global `event_timestamp_ns` monotonicity across realtime/full graphs: delayed
full-graph batches can represent an older graph frontier. Causal analysis uses
the realtime stream; `subject_timestamp_ns` may precede event time by design.
Empty optional numeric fields become `None`; required numeric fields may not be
empty.

- [ ] **Step 3: Reuse the established IMU and trajectory loaders**

Import `load_imu`, `load_mocap_trajectory`, `load_okvis_trajectory`,
`parse_image_delay`, and `correct_camera_timestamps` from
`analyze_repeatability.py`. Load `mocap_path`, `config_path`, and
`image_delay_s` from `run_manifest.json`; require the files to exist and require
the parsed config delay to match the manifest. Keep nanoseconds as integers
until the final join; convert to seconds relative to the first corrected camera
frame to avoid floating-point precision loss.

Reuse the position-only correction from `mocap_reference_correction.py`:

```python
def correct_reference_positions(
    positions: np.ndarray,
    quaternions_wxyz: np.ndarray,
    lever_m: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    quaternions = np.asarray(quaternions_wxyz, dtype=float)
    lever = np.asarray(lever_m, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must be N x 3")
    if quaternions.shape != (len(positions), 4) or lever.shape != (3,):
        raise ValueError("quaternions or lever have invalid shape")
    rotations = Rotation.from_quat(quaternions[:, [1, 2, 3, 0]])
    return positions + rotations.apply(np.broadcast_to(lever, positions.shape))
```

Keep the shared regression test proving `apply_effective_lever()` returns the
same corrected reference values. Import this helper together with
`FIXED_DIAGNOSTIC_LEVER_M` and `session_fixed_lever` in the causal analyzer, so
the existing cross-sample report and new local-motion analysis use one
correction formula without forcing a whole-trajectory alignment.

Apply `correct_camera_timestamps(raw_camera_timestamps, image_delay_s)` before
joining archived images to diagnostics because `ThreadedSlam::addImages()`
subtracts the same delay. Add a fixture with a `24.87 ms` delay and prove an
image lands on the corrected diagnostic frame, not the raw timestamp.

Associate each frame with IMU samples in the half-open interval between the
previous and current camera timestamps. Compute gyro magnitude maximum, mean,
integrated angle, sample count, maximum IMU gap and saturation count.

Load the manifest mocap trajectory and apply the established shared
20260805 rigid-body correction functions. For every camera interval and event
window compute external body-origin translation, rotation, and
`translation_m / rotation_rad` only when rotation is at least `1e-3 rad`;
otherwise leave the ratio empty. Keep these names prefixed
`mocap_body_`: they are an external motion control, not accepted-feature
parallax or triangulation baseline.

- [ ] **Step 4: Implement angular event extraction**

Use the established rule exactly:

```python
@dataclasses.dataclass(frozen=True)
class AngularEvent:
    start_s: float
    end_s: float
    peak_radps: float
    integrated_angle_rad: float


def detect_angular_events(
    timestamps_s: np.ndarray,
    angular_speed_radps: np.ndarray,
    threshold_radps: float = 3.0,
    minimum_duration_s: float = 0.05,
    merge_gap_s: float = 0.25,
) -> list[AngularEvent]:
    if timestamps_s.ndim != 1 or angular_speed_radps.shape != timestamps_s.shape:
        raise ValueError("timestamps and angular speed must be matching 1-D arrays")
    if timestamps_s.size < 2 or np.any(np.diff(timestamps_s) <= 0):
        raise ValueError("timestamps must contain at least two increasing samples")
    above = angular_speed_radps > threshold_radps
    transitions = np.diff(np.r_[False, above, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    qualified = [
        (int(start), int(stop))
        for start, stop in zip(starts, stops)
        if timestamps_s[stop] - timestamps_s[start] >= minimum_duration_s
    ]
    merged: list[tuple[int, int]] = []
    for start, stop in qualified:
        if merged and timestamps_s[start] - timestamps_s[merged[-1][1]] < merge_gap_s:
            merged[-1] = (merged[-1][0], stop)
        else:
            merged.append((start, stop))
    return [
        AngularEvent(
            start_s=float(timestamps_s[start]),
            end_s=float(timestamps_s[stop]),
            peak_radps=float(np.max(angular_speed_radps[start : stop + 1])),
            integrated_angle_rad=float(np.trapz(
                angular_speed_radps[start : stop + 1],
                timestamps_s[start : stop + 1],
            )),
        )
        for start, stop in merged
    ]
```

Tests must cover an event shorter than 50 ms, two runs merged across a 200 ms
gap, two runs kept separate across a 300 ms gap, integrated angle using
trapezoidal integration, and an event touching dataset start/end.

- [ ] **Step 5: Compute image proxies offline in event windows**

For camera frames inside `[-5,+10] s` around each angular event, compute image
statistics from archived PNG files, not in the VIO process. Also enumerate the
same-run low-angular candidate windows defined in Task 10 before matching, and
compute identical image statistics for every candidate window; caliper/ranking
must run only after both event and candidate image fields exist:

```python
def compute_image_statistics(path: Path) -> dict[str, float]:
    image = Image.open(path).convert("L")
    if image.width > 640:
        height = round(image.height * 640 / image.width)
        image = image.resize((640, height), Image.Resampling.BILINEAR)
    values = np.asarray(image, dtype=np.float64)
    center = values[1:-1, 1:-1]
    laplacian = (
        values[:-2, 1:-1] + values[2:, 1:-1]
        + values[1:-1, :-2] + values[1:-1, 2:] - 4.0 * center
    )
    gradient_y, gradient_x = np.gradient(values)
    return {
        "image_laplacian_variance": float(np.var(laplacian)),
        "image_gradient_median": float(np.median(np.hypot(gradient_x, gradient_y))),
        "image_intensity_stddev": float(np.std(values)),
    }
```

Validate images have at least `3x3` pixels and finite statistics. Tests use a
uniform image, checkerboard and Gaussian-blurred checkerboard; the blurred image
must reduce Laplacian variance and median gradient. Keep these named as image
proxies because texture and blur remain partially confounded. Add a test with
one angular event and two low-angular candidates proving all three windows
receive image statistics before control selection.

- [ ] **Step 6: Build one joined frame table**

Write `tables/causal_frame_metrics.csv` with one row per sequence/run/frame.
Join repeated initialisation, GP3P and triangulation rows by aggregation,
preserving maxima, counts and physical-unit quantiles. Add frame-interval IMU
exposure, external mocap body motion, offline image proxies, time relative to
event start, pre/event/post phase, local trajectory error against mocap, and
pre-event map-health fields.

Use the online `okvis2-*_trajectory.csv` for time-local drift onset and the
final-BA trajectory only for sequence-level APE. For each event, associate the
online trajectory to mocap, estimate one rigid SE(3) alignment using only the
healthy `[-5,-1] s` pre-event window, freeze that transform, and evaluate all
later local errors with it. Require at least 30 associated pre-event poses and
at least 2 s of coverage; otherwise mark event-local APE unavailable. Never use
the existing whole-trajectory `evaluate_ape()` alignment for drift onset,
because post-failure poses would leak future information into the pre-event
reference frame. Reuse the existing 20260805 correction map; do not introduce a
second correction table.

Reject joins where nearest timestamps differ by more than one camera period;
report unmatched counts in `tables/causal_diagnostics_coverage.csv`.

- [ ] **Step 7: Run unit tests**

```bash
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_mocap_reference_correction \
  tools.accuracy_analysis.scripts.test_analyze_vio_causal_diagnostics -v
```

Expected: all schema, event and join tests pass.

- [ ] **Step 8: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis/scripts/mocap_reference_correction.py tools/accuracy_analysis/scripts/test_mocap_reference_correction.py tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py
```

Expected: no whitespace errors; no commit.

## Task 10: Implement Temporal-Mediation and Recovery Analysis

**Files:**

- Modify: `tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py`

- [ ] **Step 1: Write failing onset/recovery tests**

Use synthetic traces with known onset order. Verify that a mediator changing at
`t=0.10`, GP3P collapsing at `t=0.20`, and map support collapsing at `t=0.40`
is accepted as temporally ordered, while a mediator changing at `t=0.30` is
classified downstream of GP3P.

Pre-register the event windows relative to angular-event start/end:

```text
baseline:       [start-5 s, start-1 s]
angular input:  [start, end]
mediator:       [start, end+0.5 s]
GP3P onset scan:[start, end+2 s]
GP3P persistence outcome: [end+0.5 s, end+2 s]
map outcome:    [end+2 s, end+5 s]
late recovery:  [end+5 s, end+10 s]
```

For frame-valued metrics, compute baseline median and
`scale=max(1.4826*MAD, 0.05*abs(median), epsilon)`, with epsilon `1` for counts,
`0.1 px` for pixel errors, `1e-3` for ratios, `1e-4 rad` for angles and
`1e-4 m` for distances. Onset is the first of three consecutive frames whose
harmful-direction robust z-score is at least 2. Recovery is the first of five
consecutive frames with absolute robust z-score at most 2. Test onset,
recovery, non-recovery, zero-MAD fallback and traces ending before the required
run length. Search GP3P onset across `[start,end+2]` so failures during the
angular event cannot be hidden by the lag window. Use the non-overlapping
post-event GP3P window only for persistence/regression outcomes. Window-valued
rates use the pre-registered windows and do not claim frame-level onset.

- [ ] **Step 2: Compute separate mediator families**

Do not create one opaque fragmentation score. Produce event-level changes for:

```python
MEDIATORS = {
    "feature_availability": [
        "keypoints_total", "grid_fraction_mean", "hull_fraction_mean",
        "image_laplacian_variance_mean", "image_gradient_median_mean",
        "image_intensity_stddev_mean"
    ],
    "map_matching": [
        "projected_eligible_map_landmarks", "descriptor_comparisons",
        "descriptor_candidates_below_threshold", "accepted_map_matches",
        "best_descriptor_distance_median", "descriptor_distance_median",
        "predicted_reprojection_error_px_median"
    ],
    "triangulation_geometry": [
        "temporal_ray_angle_p10_rad", "temporal_parallel_fraction",
        "spatial_ray_angle_p10_rad", "initialisable_fraction",
        "mocap_body_translation_m", "mocap_body_rotation_rad",
        "mocap_body_translation_per_rotation_m_per_rad",
        "rotation_only_minus_relative_pose_inlier_ratio"
    ],
    "prediction_consistency": [
        "gp3p_start_to_model_rotation_rad",
        "gp3p_start_to_model_translation_m",
        "gp3p_pre_invocation_to_model_rotation_rad",
        "gp3p_pre_invocation_to_model_translation_m"
    ],
    "map_feedback": [
        "gp3p_inlier_ratio", "visual_observation_removals",
        "active_initialised_landmarks", "landmark_births"
    ],
}
```

For each value retain physical units and pre-event normalized change. Never
replace a missing metric with zero. Report 2D-2D initialisation mediators only
for frames where that path ran; do not extrapolate them to established runtime
GP3P failures.

- [ ] **Step 3: Construct within-sequence low-angular control windows**

For each event, search the same sequence and run for non-overlapping windows of
the same duration, at least 10 s from any `>3 rad/s` event and with peak angular
speed `<1 rad/s`. Rank candidates by standardized Euclidean distance on
baseline active initialized landmarks, accepted map matches,
pre-window `mocap_body_translation_m`, pre-window
`mocap_body_rotation_rad`, image Laplacian variance and keypoint count. Retain
at most three controls that are within 25% of the event baseline for every
nonzero covariate, with absolute floors `1` for counts, `1 mm` for translation,
`0.01 rad` for rotation and `1e-3` for ratios. Do not use
translation/rotation ratio for matching: near-static denominators make it
unstable. If no candidate passes, mark the event `no_matched_control`; do not
silently use unrelated frames.

Tests must prove controls come from the same sequence/run, never overlap event
exclusion zones, satisfy the angular cap, and are chosen deterministically.

- [ ] **Step 4: Implement pre-registered models and clustered uncertainty**

Bootstrap whole sequences, not frames. With a deterministic RNG seed, sample
sequence ids with replacement, include all their events/runs, and compute 2.5,
50 and 97.5 percentiles for:

1. angular exposure -> mediator change;
2. lagged mediator -> next-window GP3P/map-support outcome;
3. standardized angular coefficient before and after adding the mediator.

Use paired event-minus-control changes. Standardize continuous columns over
valid event/control pairs and fit these exact event-level models with NumPy
least squares:

```text
M_delta = a0 + a1*angular_integral + a2*pre_map_support
          + a3*mocap_translation + a4*mocap_rotation
          + a5*pre_image_sharpness
Y       = c0 + c1*angular_integral + c2*pre_map_support
          + c3*mocap_translation + c4*mocap_rotation
          + c5*pre_image_sharpness
Y       = d0 + c_prime*angular_integral + b*M_delta
          + d2*pre_map_support + d3*mocap_translation
          + d4*mocap_rotation + d5*pre_image_sharpness
```

`M_delta` uses the mediator window; `Y` is separately the next GP3P and map
outcome windows defined in Step 1. Report `c1-c_prime` only as attenuation
consistent with mediation, not a causal indirect effect until controlled
interventions pass. Also report Spearman for monotonic exposure-response. Emit
sample count, independent sequence count, matched-control coverage, effect,
interval and direction stability. With fewer than six independent sequences,
label the model `exploratory_small_n`; with fewer rows than coefficients plus
two, return `insufficient_model_rows` instead of fitting. If the standardized
design matrix condition number exceeds `1e6`, return `collinear_design` and
report the angular/mocap-rotation correlation rather than unstable
coefficients.

- [ ] **Step 5: Implement recovery contrast**

For the designed impulse subgroup, explicitly compare:

```text
175103: mediator recovery time and absence of persistent 10 cm drift
175304: mediator deficit at +1/+3/+5/+10 s and persistent drift onset
175539: mediator deficit at +1/+3/+5/+10 s and persistent drift onset
```

Write `tables/impulse_mediator_recovery.csv` and
`tables/causal_event_metrics.csv`. Keep both runs visible; sequence medians are
secondary summaries.

- [ ] **Step 6: Generate report-ready figures**

Create:

```text
figures/impulse_mediator_timeline.png
figures/angular_to_fragmentation_mediator_paths.png
figures/mediator_onset_recovery.png
```

Every panel states x meaning, y meaning and the observed rule in Chinese.
Distinguish raw measurements from offline labels. Mark `175103/175304/175539`
with triangles and retain the existing concise anti-impact explanation.

- [ ] **Step 7: Test deterministic outputs**

Use temporary fixtures and assert table row keys, event ordering, output file
set, figure nonzero size, and identical CSV bytes across two runs with the same
seed.

```bash
python3 -m unittest tools.accuracy_analysis.scripts.test_analyze_vio_causal_diagnostics -v
```

Expected: all tests pass.

- [ ] **Step 8: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis/scripts
```

Expected: no whitespace errors; no commit.

## Task 11: Integrate With the Existing Cross-Sample Report

**Files:**

- Modify: `tools/accuracy_analysis/scripts/analyze_cross_sample_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/test_analyze_cross_sample_diagnostics.py`
- Modify after real replay: `workspace/ego2_results/202608_week1_analysis/report.md`
- Modify after real replay: `workspace/ego2_results/202608_week1_analysis/tables/cross_sample_artifact_manifest.csv`

- [ ] **Step 1: Add a failing optional-integration test**

When `--causal-diagnostics-root` is absent, assert current 24-sequence output is
byte-compatible for unchanged tables. When supplied with a complete fixture,
assert the new tables/figures are added to the manifest. When supplied with an
incomplete run, assert analysis fails rather than silently falling back.

- [ ] **Step 2: Add the optional CLI argument**

```python
parser.add_argument(
    "--causal-diagnostics-root",
    type=Path,
    default=None,
    help="Validated replay root containing structured VIO diagnostics",
)
```

Keep the existing report runnable before instrumented replays exist.

- [ ] **Step 3: Add evidence without overwriting current conclusions**

Append a causal-mediator section that answers, for each H1/H2/H3/H4:

```text
direct measurement available?
temporal precedence?
dose relation?
175103 recovery contrast?
controlled intervention available?
support level and limitation
```

Do not promote a path to “strong causal evidence” until the design document's
intervention and replication criteria are met. Keep RANSAC failure and short
landmark life described as downstream state variables. For H2, put 2D-2D
rotation-only/relative-pose evidence in an “initialisation path” row and GP3P
in a separate “runtime 3D-2D path” row; never use the former to explain a
runtime interval where it did not execute.

- [ ] **Step 4: Run existing and new Python tests**

```bash
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_analyze_cross_sample_diagnostics \
  tools.accuracy_analysis.scripts.test_analyze_vio_causal_diagnostics -v
```

Expected: existing analysis behavior remains valid and optional diagnostics
integration tests pass.

- [ ] **Step 5: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis workspace/ego2_results/202608_week1_analysis
```

Expected: no whitespace errors; no commit.

## Task 12: Execute the Phased Validation

**Files:**

- Generate only after implementation: `workspace/ego2_results/202608_causal_diagnostics/`
- Update only after evidence exists: `workspace/ego2_results/202608_week1_analysis/`

- [ ] **Step 1: Run all unit tests and affected builds**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis --target \
  okvis_common_test okvis_frontend_test okvis_ceres_test \
  okvis_ceres_diagnostics_test \
  okvis_app_synchronous okvis2x_app_synchronous -j2
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis \
  -R '^(okvis_common_test|okvis_frontend_test|okvis_ceres_test|okvis_ceres_diagnostics_test)$' \
  --output-on-failure
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_mocap_reference_correction \
  tools.accuracy_analysis.scripts.test_analyze_multiday \
  tools.accuracy_analysis.scripts.test_run_vio_diagnostics \
  tools.accuracy_analysis.scripts.test_analyze_vio_causal_diagnostics \
  tools.accuracy_analysis.scripts.test_analyze_cross_sample_diagnostics -v
```

Expected: all selected builds and tests pass.

- [ ] **Step 2: Verify disabled-mode noninterference**

Run one healthy short sequence with `OKVIS_DIAGNOSTICS_DIR` unset. Assert no
`vio_diag_*.csv` files are created. Compare final-BA trajectory against the
same source build's second disabled run using the existing repeatability
metrics; do not require byte identity because estimator scheduling may be
nondeterministic.

- [ ] **Step 3: Measure enabled-mode overhead on one healthy sequence**

Replay `20260806-174511` once disabled and once enabled. Record wall time,
diagnostic directory size, frame count and final-BA APE. Accept phase 1 only if:

```text
all six CSVs pass schema validation
metadata has run_complete=true
diagnostic frame coverage >= 99.5%
event_sequence is complete and strictly increasing
no writer failure
enabled runtime overhead <= 15%
qualitative trajectory outcome and APE class are unchanged
```

If overhead exceeds 15%, profile accumulator allocation and CSV flush frequency
before reducing required fields.

- [ ] **Step 4: Replay the impulse subgroup twice**

```bash
python3 tools/accuracy_analysis/scripts/run_vio_diagnostics.py \
  --reference-results-root workspace/ego2_results \
  --sequences 20260806-175103 20260806-175304 20260806-175539 \
  --repeats 2 --jobs 1
```

Expected: six complete run manifests. Recompute APE and confirm `175103`
recovers while `175304/175539` retain the previously observed failure classes.
If instrumentation changes those qualitative outcomes, stop before population
replay and treat instrumentation perturbation as a defect.

- [ ] **Step 5: Analyze impulse temporal ordering**

```bash
python3 tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py \
  --diagnostics-root workspace/ego2_results/202608_causal_diagnostics \
  --data-root /home/chenguyuan/data \
  --output workspace/ego2_results/202608_week1_analysis \
  --sequences 20260806-175103 20260806-175304 20260806-175539
```

Expected: coverage table, event table, recovery table and three figures are
generated. The report states which paths are time-ordered and which remain
unsupported; it does not call three sequences a population-level proof.

- [ ] **Step 6: Replay all 24 sequences only after phase 2 passes**

Discover sequences from the existing cross-sample manifest, require exactly
24 unique names, and run two repeats unless storage/runtime constraints are
explicitly changed by the user. Use `--jobs 1` for the first population replay
to avoid changing scheduling relative to prior experiments.

- [ ] **Step 7: Regenerate the unified analysis**

```bash
python3 tools/accuracy_analysis/scripts/analyze_cross_sample_diagnostics.py \
  --results-root workspace/ego2_results \
  --data-root /home/chenguyuan/data \
  --output workspace/ego2_results/202608_week1_analysis \
  --causal-diagnostics-root workspace/ego2_results/202608_causal_diagnostics
```

Expected: all 24 sequences are present, sensitivity cohorts remain available,
and the new mediator evidence is integrated without treating 20260806 as the
main population.

- [ ] **Step 8: Final verification and diff review**

```bash
git diff --check
git status --short
```

Summarize builds, tests, replay coverage, rejected runs, runtime overhead,
evidence gained and unresolved causal paths. Do not stage or commit.

## Task 13: Prepare Controlled Interventions

**Files:**

- Create: `tools/accuracy_analysis/scripts/prepare_imu_time_offset_variants.py`
- Create: `tools/accuracy_analysis/scripts/test_prepare_imu_time_offset_variants.py`
- Create: `tools/accuracy_analysis/docs/VIO_CAUSAL_EXPERIMENT_PROTOCOL.md`
- Modify: `tools/accuracy_analysis/scripts/run_vio_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py`

- [ ] **Step 1: Write failing source-preservation tests**

Create a miniature EuRoC fixture with `cam0/cam1/imu0`. Generate `-10`, `0`,
and `+10 ms` variants and assert:

```python
assert source_imu.read_bytes() == source_bytes
assert shifted_plus_10_ns - original_ns == 10_000_000
assert shifted_minus_10_ns - original_ns == -10_000_000
assert np.all(np.diff(shifted_timestamps) > 0)
assert (variant / "cam0").is_symlink()
```

Also test refusal of an existing target, a missing source camera, malformed IMU
rows and a shift that would produce a negative timestamp.

- [ ] **Step 2: Implement immutable time-offset views**

The CLI is:

```text
--source-dataset /home/chenguyuan/data/20260806/20260806-175304_euroc
--output-root workspace/ego2_results/202608_causal_diagnostics/dataset_variants
--offsets-ms -10 -5 -2 0 2 5 10
```

For every offset, create a new directory, symlink each camera directory and any
non-IMU sensor directory, create a real `imu0/data.csv` with only the integer
nanosecond timestamp shifted, and write `variant_manifest.json`. Build each
variant under a sibling temporary directory, fsync/close its manifest, then
atomically rename the complete directory to its final name. On failure, leave
no final target and report the temporary path for inspection. Do not copy or
rewrite image files and do not modify the source `.complete` marker.

- [ ] **Step 3: Let the runner consume an explicit dataset manifest**

Add mutually exclusive `--sequences` and `--dataset-manifest` modes. Manifest
rows contain `experiment_id`, `dataset`, `sequence`, `intervention`, and
`intervention_value`. Preserve these fields in every run manifest and causal
event table.

- [ ] **Step 4: Write the acquisition protocol**

The protocol fixes four matrices, with five repeated acquisitions per cell:

```text
Geometry: same scene/exposure/angular profile; 0-2 cm, 10-20 cm, >=30 cm
          camera-center displacement over the 0.5 s event window.
Image:    same scene/6-DoF motion; 0.5x, 1x, 2x nominal exposure, constrained
          to avoid clipping and recorded in microseconds.
Texture:  same motion/exposure/light; low, medium and high feature-density
          targets, validated by pre-event keypoint/grid-coverage ranges.
Timing:   archived data replay at -10/-5/-2/0/+2/+5/+10 ms IMU timestamp shift.
```

For each acquisition record `experiment_id`, operator, camera configuration
hash, scene id, motion profile id, actual exposure, illumination, translation
and rotation over the event, repeat id, raw dataset path, mocap log and any
deviation. Randomize cell order within a session to reduce warm-up/order bias.

- [ ] **Step 5: Define intervention acceptance tests**

Before causal comparison, require within each matrix:

```text
angular peak and integrated angle overlap across cells
non-target controls remain within predeclared tolerance
all five repeats have complete diagnostics and mocap
the manipulated mediator changes in the intended direction
```

Geometry cells must change accepted ray-angle/baseline metrics; exposure cells
must change image sharpness/gradient without changing the commanded motion;
timing cells must change prediction/reprojection metrics while image content is
identical. A failed manipulation check is reported as an invalid intervention,
not a negative causal result.

- [ ] **Step 6: Run tool tests and a dry-run variant build**

```bash
python3 -m unittest \
  tools.accuracy_analysis.scripts.test_prepare_imu_time_offset_variants \
  tools.accuracy_analysis.scripts.test_run_vio_diagnostics -v
python3 tools/accuracy_analysis/scripts/prepare_imu_time_offset_variants.py \
  --source-dataset /home/chenguyuan/data/20260806/20260806-175304_euroc \
  --output-root /tmp/okvis_vio_offset_dry_run \
  --offsets-ms 0
```

Expected: tests pass; the `/tmp` variant contains symlinked camera directories,
a real shifted IMU CSV and a manifest, while the source checksum is unchanged.

- [ ] **Step 7: Review without committing**

```bash
git diff --check -- tools/accuracy_analysis
git status --short
```

Expected: only planned files changed; no commit.

## Completion Criteria

The implementation is complete only when:

1. diagnostics are absent when disabled and schema-complete when enabled;
2. no frontend or graph-mutation hot loop performs filesystem writes or waits
   on the shared writer;
3. visual rejection and backend maintenance reasons are distinguishable;
4. temporal/spatial ray geometry comes from accepted feature rays, not mocap
   body translation;
5. 2D-2D rotation-only and relative-pose model support is recorded separately
   from runtime 3D-2D GP3P;
6. all GP3P invocations, including no-prior-frame and `<10 correspondence`
   early returns, are represented with frame-entry/pre-invocation/model poses;
7. landmark event, subject and birth times are distinct, and event order uses a
   stable sequence id;
8. `175103` can be compared against `175304/175539` at mediator recovery level;
9. local drift uses only a frozen pre-event alignment and camera images use the
   configured OKVIS image-delay correction;
10. Python statistics use matched within-sequence controls and cluster by
    sequence/event rather than treating frames as independent;
11. instrumented replay preserves qualitative baseline behavior;
12. the report grades every candidate middle path with limitations;
13. controlled-intervention tools preserve archived source data and enforce
    manipulation checks;
14. no commit has been created without an explicit user request.
