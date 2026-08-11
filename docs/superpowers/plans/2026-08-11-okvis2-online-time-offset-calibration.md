# OKVIS2 Online Camera-IMU Time-Offset Calibration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在保持 OKVIS2 因果在线 SLAM 流程不变的前提下，在线估计四个硬同步相机共享的相机-IMU 残余时间偏移，并把通过可观测性检查的估计反馈到后续 IMU 传播、地图投影和 GP3P 初值。

**Architecture:** 保留 YAML/Kalibr `image_delay` 作为名义值以及所有 `MultiFrame`/graph state 的名义时间戳；新增一个 realtime graph 权威、full graph 固定同步副本的全局一维 `delta_d`。IMU 因子在 `[t0_nominal-delta_d, t1_nominal-delta_d]` 上用插值边界重新预积分，前端在真正修改 observation 前用相同传播模型只读比较少量 delay hypothesis；后端以 `WARMUP/ESTIMATING/HOLD` 状态机、信息量检查和完整参数快照回滚控制更新。

**Tech Stack:** C++17, CMake, Ceres Solver, Eigen, GoogleTest, OpenCV FileStorage, existing BRISK/opengv frontend, existing bounded CSV diagnostics, Python 3 replay/analysis tools.

---

## Execution Rules

- 设计依据是 `docs/superpowers/specs/2026-08-11-okvis2-online-time-offset-calibration-design.md`。符号固定为 `t_effective = t_nominal - delta_d`；正 `delta_d` 表示曝光早于当前名义校正时间。
- SLAM 始终因果在线：frame `k` 的 BA 结果只反馈到 `k+1` 及以后；frame `k` 只能在自身 observation 写入前使用由历史状态、当前图像特征和截至当前时刻 IMU 构成的只读 hypothesis。
- 允许 EuRoC 本地回放慢于墙钟时间，但禁止读取未来帧、全序列时间偏移搜索或 SfM 式全量优化。
- `39.25 ms` 是单序列过拟合结果，不出现在默认值、测试真值、候选中心或验收标准中。
- “小改动”指保留 preprocessing、BRISK、map matching、GP3P、BA graph、outlier removal、landmark cleanup 和 loop closure 的现有职责与顺序；为正确建模允许增加聚焦 helper、第五个 IMU 参数块和必要 API。
- 不修改现有视觉 reprojection factor 的三个参数块，不加入 feature velocity、per-camera delay、clock skew、random walk、rolling shutter 或 final-BA delay 释放。
- 所有新增功能默认关闭。`do_time_offset: false` 时 `delta_d=0`、前端不创建 hypothesis，输出的残差/Jacobian/轨迹/匹配决策必须与旧路径在测试容差内一致。
- 不把时间偏移诊断写入 `vio_diag_landmark_events.csv`。每帧只写有界数量的 hypothesis 和一条 calibration record。
- 当前工作区已有大量用户改动。执行时逐文件增量合并，不覆盖、不还原、不格式化无关代码。
- 用户未授权 commit。本计划故意不包含 `git add`、`git commit`、分支合并或历史改写步骤；每个任务以定向测试和 `git diff --check` 收尾。

## File Map

Create:

- `okvis_ceres/include/okvis/ceres/TimeOffsetParameterBlock.hpp`: 一维秒单位参数块。
- `okvis_ceres/src/TimeOffsetParameterBlock.cpp`: 参数存储、plus/minus 和固定状态实现。
- `okvis_ceres/include/okvis/ceres/TimeOffsetError.hpp`: 一维高斯先验 cost function。
- `okvis_ceres/src/TimeOffsetError.cpp`: 先验 residual/Jacobian。
- `okvis_ceres/include/okvis/ceres/ImuPreintegration.hpp`: 有效时间、插值边界、coverage 与 `PreintegrationResult` 公共契约。
- `okvis_ceres/src/ImuPreintegration.cpp`: 从现有 `ImuError` 提取的无副作用移动边界预积分。
- `okvis_ceres/include/okvis/TimeOffsetCalibration.hpp`: 状态机、观测质量、候选、验收/拒绝类型。
- `okvis_ceres/src/TimeOffsetCalibration.cpp`: 候选生成、信息量判定、稳定计数和状态转移。
- `okvis_ceres/test/TestTimeOffsetParameter.cpp`: parameter block 与 prior 测试。
- `okvis_ceres/test/TestShiftedImuPreintegration.cpp`: 符号、插值、coverage、常速不可观测试。
- `okvis_ceres/test/TestImuTimeOffsetError.cpp`: 第五参数块、numeric Jacobian 和 synthetic recovery 测试。
- `okvis_ceres/test/TestTimeOffsetGraph.cpp`: graph 生命周期、merge、sync、covariance、rollback 测试。
- `okvis_ceres/test/TestTimeOffsetCalibration.cpp`: 状态机和验收规则测试。
- `okvis_frontend/include/okvis/TimeOffsetHypothesis.hpp`: hypothesis pose、score、selection 与当前帧视觉支撑结构。
- `okvis_frontend/src/TimeOffsetHypothesis.cpp`: 稳定字典序排序和 tie handling。
- `okvis_frontend/test/TestTimeOffsetHypothesis.cpp`: 排序、只读和多相机门限测试。
- `okvis_multisensor_processing/include/okvis/TimeOffsetTemporalCoverage.hpp`: IMU 周期、所需 overlap 和 coverage 的纯函数。
- `okvis_multisensor_processing/src/TimeOffsetTemporalCoverage.cpp`: 不依赖线程/队列状态的 coverage 计算。
- `okvis_multisensor_processing/test/TestTimeOffsetTemporalCoverage.cpp`: 动态 overlap、无外推和当前帧接线测试。

Modify:

- `okvis_common/include/okvis/Parameters.hpp`: 向 `OnlineCalibrationParameters` 增加带默认值的 time-offset 配置。
- `okvis_common/src/ViParametersReader.cpp`: 可选字段解析、范围检查和 IMU 前置条件。
- `okvis_common/test/TestViParametersReader.cpp`: 缺省兼容、显式解析和非法配置测试。
- `okvis_common/CMakeLists.txt`: 注册 reader 测试源。
- `okvis_ceres/include/okvis/ceres/ImuError.hpp`: IMU/Pseudo-IMU cost function 第五块、offset-aware cache 和传播重载。
- `okvis_ceres/src/ImuError.cpp`: 调用纯预积分 helper、计算中心差分时间 Jacobian、clone/sync 新状态。
- `okvis_ceres/include/okvis/ViGraph.hpp`: 全局 block/prior、cost/covariance、snapshot、coverage API。
- `okvis_ceres/src/ViGraph.cpp`: graph 创建、传播、clone 与 IMU residual 的第五块接线。
- `okvis_ceres/src/ViGraphEstimator.cpp`: state elimination/IMU merge 后重建第五块 residual。
- `okvis_ceres/src/Component.cpp`: component/map 载入时创建同一五块 IMU residual。
- `okvis_ceres/include/okvis/ViSlamBackend.hpp`: realtime 权威值、候选启动、视觉支撑提交和 diagnostics getters。
- `okvis_ceres/src/ViSlamBackend.cpp`: controller、full graph 固定副本、staged extrinsics、验收和重新求解回滚。
- `okvis_ceres/CMakeLists.txt`: 编译新 helper 并注册五个测试源。
- `okvis_multisensor_processing/include/okvis/ThreadedSlam.hpp`: 缓存 accepted offset、动态 overlap 和 hypothesis 接线 helper。
- `okvis_multisensor_processing/src/ThreadedSlam.cpp`: 所有 IMU window/propagation 使用有效边界，当前帧在 `addStates()` 前选择 hypothesis。
- `okvis_multisensor_processing/CMakeLists.txt`: 注册 coverage 测试。
- `okvis_frontend/include/okvis/Frontend.hpp`: 暴露只读 scorer 和上一帧视觉支撑，提取非写入 candidate collection。
- `okvis_frontend/src/Frontend.cpp`: 复用现有投影/BRISK/GP3P 逻辑评分，真正 map matching 仍只执行一次。
- `okvis_frontend/CMakeLists.txt`: 编译 hypothesis helper 并注册测试。
- `okvis_common/include/okvis/VioDiagnostics.hpp`: 有界 `TimeOffsetDiagnosticRecord` 和 writer API。
- `okvis_common/src/VioDiagnostics.cpp`: `vio_diag_time_offset.csv` schema/header/row。
- `okvis_common/test/TestVioDiagnostics.cpp`: time-offset CSV 宽度、空值和有界 hypothesis 测试。
- `tools/accuracy_analysis/scripts/run_vio_diagnostics.py`: 新 schema 的条件验证和 manifest 配置记录。
- `tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py`: 旧 schema 兼容和新 CSV 必需性测试。
- `tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py`: 汇总 offset、sigma、状态转移及视觉支撑的时间先后关系。
- `tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py`: 新诊断解析和 causal-order 测试。
- `config/okvis2_eucm_EGO2.yaml`: 显式加入默认关闭的 time-offset 参数；保持现有 `image_delay` 数值不变。

## Stable Contracts

以下类型和签名在任务间保持一致，实施中不要为局部方便另造同义接口：

```cpp
namespace okvis {

inline Time effectiveTime(const Time& nominal, double deltaSeconds) {
  return nominal - Duration(deltaSeconds);
}

namespace ceres {

class TimeOffsetParameterBlock
    : public ParameterBlockSized<1, 1, double> {
 public:
  explicit TimeOffsetParameterBlock(double deltaSeconds = 0.0,
                                    uint64_t id = 0);
  void setParameters(const double* parameters) override;
  double* parameters() override;
  const double* parameters() const override;
  void plus(const double* x, const double* delta, double* out) const override;
  void plusJacobian(const double*, double* jacobian) const override;
  void minus(const double* x, const double* y, double* delta) const override;
  void liftJacobian(const double*, double* jacobian) const override;
  std::string typeInfo() const override { return "TimeOffsetParameterBlock"; }
};

struct PreintegrationResult {
  bool valid = false;
  int integrationSteps = 0;
  Time effectiveStart;
  Time effectiveEnd;
  Eigen::Quaterniond Delta_q = Eigen::Quaterniond::Identity();
  Eigen::Matrix3d C_integral = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d C_doubleintegral = Eigen::Matrix3d::Zero();
  Eigen::Vector3d acc_integral = Eigen::Vector3d::Zero();
  Eigen::Vector3d acc_doubleintegral = Eigen::Vector3d::Zero();
  Eigen::Matrix3d cross = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dalpha_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dv_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix3d dp_db_g = Eigen::Matrix3d::Zero();
  Eigen::Matrix<double, 15, 15> covariance =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> information =
      Eigen::Matrix<double, 15, 15>::Zero();
  Eigen::Matrix<double, 15, 15> squareRootInformation =
      Eigen::Matrix<double, 15, 15>::Zero();
  AlignedVector<Eigen::Matrix<double, 15, 15>> covarianceNoiseTerms;
  std::string invalidReason;
};

PreintegrationResult preintegrateImu(
    const ImuMeasurementDeque& measurements,
    const ImuParameters& parameters,
    const Time& nominalStart,
    const Time& nominalEnd,
    double deltaSeconds,
    const SpeedAndBias& referenceSpeedAndBias);

}  // namespace ceres

enum class TimeOffsetCalibrationState { Warmup, Estimating, Hold };

struct TimeOffsetVisualSupport {
  size_t acceptedInitialisedMatches = 0;
  size_t contributingCameras = 0;
  bool primaryRansacAttempted = false;
  bool primaryRansacSucceeded = false;
  bool imuCoverageValid = true;
};

struct TimeOffsetPoseHypothesis {
  double deltaSeconds = 0.0;
  kinematics::Transformation T_WS;
  SpeedAndBias speedAndBias = SpeedAndBias::Zero();
};

struct TimeOffsetHypothesisScore {
  double deltaSeconds = 0.0;
  size_t gp3pInliers = 0;
  size_t acceptedInitialisedMatches = 0;
  size_t contributingCameras = 0;
  std::optional<double> medianPredictedReprojectionErrorPx;
  bool gp3pComputed = false;
  bool coverageValid = true;
};

struct TimeOffsetSelection {
  double deltaSeconds = 0.0;
  kinematics::Transformation T_WS;
  SpeedAndBias speedAndBias = SpeedAndBias::Zero();
  std::vector<TimeOffsetHypothesisScore> scores;
  bool changedFromAccepted = false;
};

}  // namespace okvis
```

Controller 的观测输入和决定固定如下：

```cpp
struct TimeOffsetUpdateEvidence {
  bool solverUsable = false;
  bool finiteEstimate = false;
  bool strictlyInsideBounds = false;
  bool allImuFactorsValid = false;
  double candidateSeconds = 0.0;
  double posteriorSigmaSeconds = std::numeric_limits<double>::infinity();
  double costBefore = std::numeric_limits<double>::infinity();
  double costAfter = std::numeric_limits<double>::infinity();
  TimeOffsetVisualSupport visualSupport;
};

struct TimeOffsetUpdateDecision {
  bool releaseForSolve = false;
  bool acceptCandidate = false;
  bool rerunWithAcceptedFixed = false;
  bool fixExtrinsics = true;
  TimeOffsetCalibrationState nextState = TimeOffsetCalibrationState::Warmup;
  std::string reason;
};
```

## Task 1: Add Backward-Compatible Configuration

**Files:**

- Modify: `okvis_common/include/okvis/Parameters.hpp`
- Modify: `okvis_common/src/ViParametersReader.cpp`
- Create: `okvis_common/test/TestViParametersReader.cpp`
- Modify: `okvis_common/CMakeLists.txt`

- [ ] **Step 1: Write failing default and explicit parsing tests**

Add `TestViParametersReader.cpp` to the `okvis_common_test` source list first, then add three GoogleTests that load the existing EGO2 YAML through `ViParametersReader`: the untouched file must produce `do_time_offset == false` and the defaults below; a temporary copy with all new keys inserted must reproduce exact values; a copy with `do_time_offset: true` and `imu_parameters.use: false` must throw `ViParametersReader::Exception`.

```cpp
EXPECT_FALSE(p.camera.online_calibration.do_time_offset);
EXPECT_DOUBLE_EQ(p.camera.online_calibration.time_offset_prior_sigma, 0.005);
EXPECT_DOUBLE_EQ(p.camera.online_calibration.time_offset_bound, 0.020);
EXPECT_DOUBLE_EQ(
    p.camera.online_calibration.time_offset_numeric_diff_epsilon, 1.0e-5);
EXPECT_EQ(p.camera.online_calibration.time_offset_hypothesis_count, 9);
EXPECT_EQ(
    p.camera.online_calibration.time_offset_min_initialized_matches, 30);
EXPECT_EQ(p.camera.online_calibration.time_offset_min_cameras, 2);
EXPECT_DOUBLE_EQ(
    p.camera.online_calibration.time_offset_max_posterior_sigma, 0.004);
EXPECT_EQ(
    p.camera.online_calibration.time_offset_stable_update_count, 5);
```

Use `OKVIS_TEST_SOURCE_DIR` from CMake to locate `config/okvis2_eucm_EGO2.yaml`; write modified copies under `testing::TempDir()` and never edit the source config from a test. Add this exact test-only definition:

```cmake
target_compile_definitions(okvis_common_test PRIVATE
  OKVIS_TEST_SOURCE_DIR="${PROJECT_SOURCE_DIR}")
```

- [ ] **Step 2: Run the reader test and verify the missing members fail compilation**

Run:

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_common_test
```

Expected: compilation fails because `OnlineCalibrationParameters` has no `do_time_offset` member.

- [ ] **Step 3: Add initialized configuration members**

Append these exact fields to `CameraParameters::OnlineCalibrationParameters`; the in-class initializers are what preserve behavior when YAML keys are absent.

```cpp
bool do_time_offset = false;
double time_offset_prior_sigma = 0.005;
double time_offset_bound = 0.020;
double time_offset_numeric_diff_epsilon = 1.0e-5;
int time_offset_hypothesis_count = 9;
int time_offset_min_initialized_matches = 30;
int time_offset_min_cameras = 2;
double time_offset_max_posterior_sigma = 0.004;
int time_offset_stable_update_count = 5;
```

- [ ] **Step 4: Parse only present keys and validate the complete contract**

In `ViParametersReader::readConfigFile()`, check each new `cv::FileNode` with `empty()` before calling the existing typed `parseEntry()`. After IMU parsing, throw `ViParametersReader::Exception` unless all of these hold:

```cpp
priorSigma > 0.0;
bound > 0.0;
numericDiffEpsilon > 0.0;
numericDiffEpsilon < bound;
hypothesisCount >= 1 && (hypothesisCount % 2 == 1);
minInitializedMatches >= 1;
minCameras >= 1;
maxPosteriorSigma > 0.0;
stableUpdateCount >= 1;
!doTimeOffset || viParameters_.imu.use;
```

Reject non-finite doubles. The bound is on `delta_d`, not absolute `image_delay`; do not modify `image_delay` in this parser.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_common_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_common_test$' --output-on-failure
```

Expected: `okvis_common_test` passes, including missing-key, explicit-key and IMU-disabled cases.

## Task 2: Add the Scalar Parameter Block and Gaussian Prior

**Files:**

- Create: `okvis_ceres/include/okvis/ceres/TimeOffsetParameterBlock.hpp`
- Create: `okvis_ceres/src/TimeOffsetParameterBlock.cpp`
- Create: `okvis_ceres/include/okvis/ceres/TimeOffsetError.hpp`
- Create: `okvis_ceres/src/TimeOffsetError.cpp`
- Create: `okvis_ceres/test/TestTimeOffsetParameter.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Write failing parameter/prior tests**

Add `TestTimeOffsetParameter.cpp` to `okvis_ceres_test` before building. Cover zero construction, ID/fixed metadata, `plus`, `minus`, both 1x1 Jacobians and the prior sign convention:

```cpp
TimeOffsetParameterBlock block(0.003, 42);
double step = -0.001;
double out = 0.0;
block.plus(block.parameters(), &step, &out);
EXPECT_DOUBLE_EQ(out, 0.002);
block.minus(block.parameters(), &out, &step);
EXPECT_DOUBLE_EQ(step, -0.001);

TimeOffsetError prior(0.0, 0.005);
const double* parameters[] = {block.parameters()};
double residual = 0.0;
double jacobian = 0.0;
double* jacobians[] = {&jacobian};
ASSERT_TRUE(prior.Evaluate(parameters, &residual, jacobians));
EXPECT_NEAR(residual, 0.6, 1.0e-12);
EXPECT_NEAR(jacobian, 200.0, 1.0e-12);
```

- [ ] **Step 2: Build to prove the new types are absent**

Run the `okvis_ceres_test` target. Expected: missing-header failure.

- [ ] **Step 3: Implement the scalar parameter block**

Derive from `ParameterBlockSized<1, 1, double>`, initialize `estimate_`, `id_` and `fixed_` in both constructors, return `&estimate_` from both `parameters()` overloads, and implement Euclidean `plus/minus`; both `plusJacobian()` and `liftJacobian()` return the 1x1 identity. Do not add a Ceres manifold for this scalar.

- [ ] **Step 4: Implement the scalar prior error**

Derive from `::ceres::SizedCostFunction<1, 1>` and `ErrorInterface`. Define:

```cpp
residual[0] = (parameters[0][0] - meanSeconds_) / sigmaSeconds_;
jacobians[0][0] = 1.0 / sigmaSeconds_;
```

Reject non-positive/non-finite sigma in the constructor. `EvaluateWithMinimalJacobians()` writes the same 1x1 Jacobian to both requested outputs and `typeInfo()` returns `"TimeOffsetError"`.

- [ ] **Step 5: Register sources and run tests**

Add both headers/sources to the `okvis_ceres` library; the failing-test step has already registered `TestTimeOffsetParameter.cpp` in `okvis_ceres_test`.

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_ceres_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_ceres_test$' --output-on-failure
```

Expected: scalar and prior tests pass.

## Task 3: Extract Pure Shifted IMU Preintegration

**Files:**

- Create: `okvis_ceres/include/okvis/ceres/ImuPreintegration.hpp`
- Create: `okvis_ceres/src/ImuPreintegration.cpp`
- Create: `okvis_ceres/test/TestShiftedImuPreintegration.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Write failing effective-time and coverage tests**

Add `TestShiftedImuPreintegration.cpp` to `okvis_ceres_test` before building. Use 1 kHz synthetic IMU samples. Assert positive offset shifts both endpoints earlier, negative offset shifts later, exact boundary samples are accepted, and a missing sample on either side returns `valid == false` rather than extrapolating.

```cpp
const auto result = preintegrateImu(
    measurements, params, Time(10.0), Time(10.1), 0.004, speedAndBias);
ASSERT_TRUE(result.valid);
EXPECT_NEAR(result.effectiveStart.toSec(), 9.996, 1.0e-12);
EXPECT_NEAR(result.effectiveEnd.toSec(), 10.096, 1.0e-12);
EXPECT_EQ(result.integrationSteps, 100);
```

- [ ] **Step 2: Write failing analytic-signal tests**

Generate constant and linearly varying gyro/acceleration signals. Check interpolation values at non-sample boundaries, legacy `delta_d=0` increments, positive/negative shifts, finite covariance, and the invariant that shifting a constant waveform changes neither residual-producing increments nor information.

- [ ] **Step 3: Build to verify missing helper failure**

Build `okvis_ceres_test`. Expected: missing `ImuPreintegration.hpp`.

- [ ] **Step 4: Move the existing integration body into a pure function**

Copy the numerical scheme currently inside `ImuError::redoPreintegration()` into `preintegrateImu()`, but write only to a local `PreintegrationResult`. Locate bracketing samples with `lower_bound`, linearly interpolate gyro and accelerometer at both effective endpoints, retain all current saturation weighting, bias Jacobian and covariance calculations, and return these exact invalid reasons:

```text
empty_measurements
non_increasing_interval
missing_start_boundary
missing_end_boundary
non_monotonic_imu
non_finite_integration
```

The helper must never clamp or extrapolate an endpoint. `delta_d=0` must use the same trapezoidal integration equations as the legacy implementation.

- [ ] **Step 5: Add a shifted propagation wrapper**

Declare and implement:

```cpp
bool propagateImu(
    const ImuMeasurementDeque& measurements,
    const ImuParameters& parameters,
    kinematics::Transformation& T_WS,
    SpeedAndBias& speedAndBias,
    const Time& nominalStart,
    const Time& nominalEnd,
    double deltaSeconds,
    Eigen::Matrix<double, 15, 15>* covariance = nullptr,
    Eigen::Matrix<double, 15, 15>* jacobian = nullptr,
    std::string* invalidReason = nullptr);
```

It consumes `PreintegrationResult`, updates pose/speed only on success, and leaves all inputs untouched on failure.

- [ ] **Step 6: Run focused tests and compare zero-shift legacy output**

Run `okvis_ceres_test`; expected: all shifted-preintegration tests pass and the existing `TestImuError` zero-offset propagation assertions remain unchanged.

## Task 4: Extend ImuError With the Fifth Time-Offset Block

**Files:**

- Modify: `okvis_ceres/include/okvis/ceres/ImuError.hpp`
- Modify: `okvis_ceres/src/ImuError.cpp`
- Create: `okvis_ceres/test/TestImuTimeOffsetError.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Write failing residual-block shape and disabled-equivalence tests**

Add `TestImuTimeOffsetError.cpp` to `okvis_ceres_test` before building. Assert `ImuErrorBase::parameterBlocks() == 5`, dimensions are `7,9,7,9,1`, and `delta_d=0` matches a frozen pre-change golden residual plus all four existing minimal Jacobians. Update `PseudoImuError` tests to expect a fifth block with an exactly zero 15x1 Jacobian.

- [ ] **Step 2: Write a failing time-Jacobian sweep**

For angular acceleration and time-varying acceleration, request Jacobian block 4 and compare it to a high-accuracy residual central difference. Sweep `epsilon` over `5e-6, 1e-5, 2e-5`; require relative derivative variation below 2% and absolute comparison tolerance `1e-4`.

- [ ] **Step 3: Change the sized cost-function contract**

Change both `ImuErrorBase::base_t` declarations to:

```cpp
::ceres::SizedCostFunction<15, 7, 9, 7, 9, 1>
```

Keep Pseudo-IMU polymorphic by accepting and ignoring block 4. Calibration with IMU disabled is already rejected by Task 1, but the zero block keeps generic graph residual creation uniform.

- [ ] **Step 4: Replace mutable field-by-field integration with a result cache**

Store `mutable PreintegrationResult preintegration_`, `mutable double timeOffsetRefSeconds_`, `double numericDiffEpsilonSeconds_`, and `mutable std::atomic_bool lastCoverageValid_`. Change the method to `redoPreintegration(const Transformation&, const SpeedAndBias&, double deltaSeconds) const`; `EvaluateWithMinimalJacobians()` extracts `parameters[4][0]` and passes it in. Cache reuse requires both biases and time offset to match the reference.

Factor the existing residual assembly into:

```cpp
bool evaluateFromPreintegration(
    const PreintegrationResult& preintegration,
    double const* const* parameters,
    double* residuals,
    double** poseAndBiasJacobians,
    double** poseAndBiasJacobiansMinimal) const;
```

It must not read or write the factor cache.

- [ ] **Step 5: Compute the time Jacobian from independent results**

For block 4, compute `r_plus` and `r_minus` from independent calls at `delta +/- epsilon`, including each result's own square-root information, then set:

```cpp
J_delta.col(0) = (r_plus - r_minus) / (2.0 * epsilon);
```

If either side lacks coverage, return `false`, set `lastCoverageValid_ = false`, and never reuse a one-sided derivative. Guard the shared main cache with the existing mutex; the two local results require no shared mutation.

- [ ] **Step 6: Update construction, append, clone and sync**

Add `numericDiffEpsilonSeconds` to the `ImuError` constructor with default `1e-5`; preserve it and `timeOffsetRefSeconds_` in `append()`, `clone()` and `syncFrom()`. Append/merge must retain the full measurement union, including samples required at both residual bounds; it must not discard measurements based only on nominal endpoints.

- [ ] **Step 7: Run factor tests**

Run `okvis_ceres_test`. Expected: existing IMU tests, block-shape test, Pseudo-IMU zero Jacobian and epsilon sweep pass under the repository warning flags.

## Task 5: Own the Global Block in ViGraph and Preserve It Through Graph Operations

**Files:**

- Modify: `okvis_ceres/include/okvis/ViGraph.hpp`
- Modify: `okvis_ceres/src/ViGraph.cpp`
- Modify: `okvis_ceres/src/ViGraphEstimator.cpp`
- Modify: `okvis_ceres/src/Component.cpp`
- Create: `okvis_ceres/test/TestTimeOffsetGraph.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Write failing graph ownership tests**

Add `TestTimeOffsetGraph.cpp` to `okvis_ceres_test` before building. Construct a graph, configure time offset, add two states and assert one global parameter block/prior exists; every IMU residual references the same scalar pointer. Remove/merge states and assert the block remains registered and the merged residual still has five blocks. Call `ViGraphEstimator::clear()`, add states again, and assert the scalar, bounds and prior were re-registered in the newly allocated Ceres problem. Load a serialized `Component` containing an `EDGE_IMU` and assert its reconstructed full-graph residual also uses the configured scalar as block 4.

- [ ] **Step 2: Write failing snapshot, cost and covariance tests**

Perturb every variable parameter block after `captureParameterSnapshot()`, restore it, and compare byte-for-byte finite values. Verify `evaluateCost()` excludes no residuals, `timeOffsetPosteriorSigma()` is finite for an observable synthetic graph, and a fixed block reports no data covariance instead of a fake zero sigma.

- [ ] **Step 3: Add graph-level ownership and API**

Add these members and methods:

```cpp
struct ParameterSnapshot {
  std::vector<std::pair<double*, std::vector<double>>> values;
};

void configureTimeOffset(const CameraParameters::OnlineCalibrationParameters& p);
double timeOffset() const;
void setTimeOffset(double seconds);
void setTimeOffsetFixed(bool fixed);
bool timeOffsetFixed() const;
std::optional<double> timeOffsetPosteriorSigma() const;
bool allImuTimeOffsetEvaluationsValid() const;
double evaluateCost() const;
ParameterSnapshot captureParameterSnapshot() const;
void restoreParameterSnapshot(const ParameterSnapshot& snapshot);
void setOnlineExtrinsicsFixed(bool fixed);

std::shared_ptr<ceres::TimeOffsetParameterBlock> timeOffset_;
GraphEdge<ceres::TimeOffsetError> timeOffsetPrior_;
CameraParameters::OnlineCalibrationParameters onlineCalibrationParameters_;
```

`configureTimeOffset()` adds one block, prior, symmetric Ceres bounds and initially fixes the block at zero. It is idempotent only for identical parameters; conflicting second configuration throws.

Add a protected `rebuildTimeOffsetProblemState()` used by both initial configuration and `ViGraphEstimator::clear()`. `clear()` allocates a new `ceres::Problem`, then re-adds the existing scalar pointer, its prior, bounds and fixed/variable state; it must not leave a stale residual ID from the destroyed problem.

- [ ] **Step 4: Attach block 4 to every IMU/Pseudo-IMU residual**

Update `addStatesPropagate()`, `addStatesFromOther()`, every re-add path in `eliminateStateByImuMerge()`, and direct `EDGE_IMU` reconstruction in `Component.cpp` to pass `timeOffset_->parameters()` after the four state blocks. `Component` files do not serialize a second delay estimate in v1; imported IMU edges use the receiving full graph's fixed accepted scalar and configured numeric-difference epsilon. Propagation uses the graph's current scalar through `propagateImu()`. The state timestamp and `AnyState.timestamp` remain nominal.

- [ ] **Step 5: Implement safe covariance and snapshot operations**

Use `::ceres::Covariance` on `{timeOffset_->parameters(), timeOffset_->parameters()}` only when the block is variable and the last solve is usable. Return `nullopt` for fixed, singular or failed covariance. Snapshot all parameter blocks returned by `problem_->GetParameterBlocks()` with their sizes; restore only if pointer and size are still registered, otherwise throw before partial restoration.

- [ ] **Step 6: Run graph and elimination tests**

Run:

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_ceres_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_ceres_test$' --output-on-failure
```

Expected: graph ownership, elimination, component reconstruction, snapshot and covariance tests pass without changing reprojection residual block dimensions.

## Task 6: Implement Backend State Machine, Staged Extrinsics, Acceptance and Rollback

**Files:**

- Create: `okvis_ceres/include/okvis/TimeOffsetCalibration.hpp`
- Create: `okvis_ceres/src/TimeOffsetCalibration.cpp`
- Create: `okvis_ceres/test/TestTimeOffsetCalibration.cpp`
- Modify: `okvis_ceres/include/okvis/ViSlamBackend.hpp`
- Modify: `okvis_ceres/src/ViSlamBackend.cpp`
- Modify: `okvis_ceres/CMakeLists.txt`

- [ ] **Step 1: Write failing pure-controller tests**

Add `TestTimeOffsetCalibration.cpp` to `okvis_ceres_test` before building. Cover all transitions and rejection reasons: warmup lacks map support; observable healthy support enters estimating; missing coverage, repeated primary RANSAC failure or low map support enters hold; five stable accepted updates enter hold; later informative support reopens estimating. Verify prior-only posterior information is zero using:

```cpp
const double dataInformation = std::max(
    0.0, 1.0 / square(posteriorSigma) - 1.0 / square(priorSigma));
```

Constant/static windows must never release the scalar merely because the prior gives finite covariance.

Use three consecutive attempted primary-RANSAC failures as the fixed v1 transition threshold; declare it once as `static constexpr size_t kPrimaryRansacFailureHoldCount = 3` in the controller.

- [ ] **Step 2: Write failing acceptance/rollback integration tests**

Inject one accepted update and then one candidate with increased cost, bound contact, non-finite estimate, excessive posterior sigma and invalid IMU coverage. Assert each rejection restores every snapped state/landmark/speed-bias value, sets scalar to `lastAcceptedDelta`, fixes it, and invokes a second optimization exactly once. Also call the wrapper three times with `finalizeTimeOffsetUpdate=false` (matching the existing frontend refinement calls) and assert controller state, stable count and accepted scalar never change and no tentative state is copied to `fullGraph_`. Finally call `ViSlamBackend::clear()` after a configured graph, verify the controller deliberately resets to `WARMUP`/zero residual, and initialize/add a second frame successfully with both rebuilt graph scalar blocks.

- [ ] **Step 3: Implement controller decisions**

`TimeOffsetCalibrationController` owns only scalar policy data: configuration, state, last accepted value, posterior sigma, stable-update count, consecutive primary failures and reason. It does not own a Ceres problem. Candidate values are:

```cpp
std::vector<double> hypotheses() const;
TimeOffsetUpdateDecision beforeSolve(const TimeOffsetVisualSupport&);
TimeOffsetUpdateDecision afterSolve(const TimeOffsetUpdateEvidence& evidence);
double lastAcceptedDelta() const;
std::optional<double> posteriorSigma() const;
TimeOffsetCalibrationState state() const;
```

In `WARMUP`, return only `lastAcceptedDelta`; no initialized-map evidence exists yet. In `ESTIMATING`, before covariance exists, return evenly spaced values including both bounds and zero. After covariance exists, include accepted value and clipped `+/-1 sigma`, `+/-2 sigma`, then deterministically fill to the odd configured count. In `HOLD`, return only the accepted value until an actual-path healthy informative frame reopens estimation for the following frame. Deduplicate at `1e-9` seconds and sort ascending.

An accepted update counts as stable only when `abs(candidate-lastAcceptedDelta) <= posteriorSigma`; any larger accepted change resets the consecutive-stability count to one.

- [ ] **Step 4: Make realtime authoritative and full graph fixed**

Configure both graphs from the shared camera calibration after IMU registration. In the `ThreadedSlam` constructor, call `configureTimeOffsetCalibration(parameters_.camera.online_calibration)` once after the existing `addImu()` and four-camera `addCamera()` loop; do not infer four independent parameter sets. `realtimeGraph_` may release its scalar only in `ESTIMATING`; `fullGraph_` is always fixed. Add:

```cpp
double acceptedTimeOffset() const;
std::vector<double> timeOffsetHypotheses() const;
void beginTimeOffsetCandidate(double deltaSeconds);
void setTimeOffsetVisualSupport(const TimeOffsetVisualSupport& support);
TimeOffsetCalibrationState timeOffsetCalibrationState() const;
void configureTimeOffsetCalibration(
    const CameraParameters::OnlineCalibrationParameters& parameters);

std::mutex fullGraphTimeOffsetMutex_;
std::optional<double> pendingFullGraphTimeOffset_;
void queueAcceptedTimeOffsetForFullGraph(double deltaSeconds);
void applyPendingTimeOffsetToFullGraph();
```

Extend the existing optimization signature by one trailing, backward-compatible flag:

```cpp
void optimiseRealtimeGraph(
    int numIter, std::vector<StateId>& updatedStates,
    int numThreads = 1, bool verbose = false,
    bool onlyNewestState = false, bool isInitialised = true,
    bool finalizeTimeOffsetUpdate = false);
```

Existing frontend and GPS calls keep the default `false`: while a current-frame candidate is pending they optimize with that scalar fixed, do not transition the controller, do not accept/reject it, and defer realtime-to-full state/IMU imports. The single post-association call in `ThreadedSlam::optimisePublishMarginalise()` passes `true`. When no candidate is pending or calibration is disabled, auxiliary calls retain their existing full-graph behavior.

The realtime thread never writes the full scalar without `fullGraphTimeOffsetMutex_`. Hold this mutex across `optimiseFullGraph()` (including setting/clearing its ownership flags), across `synchroniseRealtimeAndFullGraph()`, and around normal-path full-graph `addStatesPropagate()` plus realtime-to-full state/IMU imports. An accepted realtime solve calls `queueAcceptedTimeOffsetForFullGraph()`; under the mutex it either applies immediately when the full graph is idle, or stores the newest accepted value. `applyPendingTimeOffsetToFullGraph()` runs under the same mutex before backlog states/IMU factors are constructed and before final BA. This may block replay while full optimization finishes, which is acceptable; it prevents mutating a Ceres parameter during a full-graph solve.

`beginTimeOffsetCandidate()` validates bounds, changes realtime only, and records a pending candidate. During `addStates()`, realtime propagates with the pending candidate while full graph propagates with the last accepted value. After an accepted solve, queue `lastAcceptedDelta`; apply it under full-graph exclusion before the existing per-frame state/IMU-factor synchronization. When loop closure owns or has an unapplied result for the full graph, retain the queued value and apply it inside `synchroniseRealtimeAndFullGraph()` before draining `addStatesBacklog_`. Never queue a rejected candidate and never invalidate timestamps.

`ViSlamBackend::clear()` first resets the controller, pending candidate and pending full-graph update to `WARMUP` with `lastAcceptedDelta=0`, sets the realtime scalar to zero, then acquires `fullGraphTimeOffsetMutex_`, sets the full scalar to zero and calls both graph `clear()` methods so their new Ceres problems re-register fixed zero blocks. This deliberate reset matches first-initialization semantics and prevents a pre-clear posterior from being reused without a map. Preserve the existing precondition that callers stop/join active optimization before `clear()`; add an assertion that `isLoopClosing_` is false.

- [ ] **Step 5: Stage time offset against existing extrinsic calibration**

In `WARMUP` and `ESTIMATING`, call `setOnlineExtrinsicsFixed(true)` for every originally configured `do_extrinsics` camera. In stable `HOLD`, fix delay and restore exactly the prior online-extrinsic policy. Reopening delay estimation fixes extrinsics before releasing delay. The existing `onlyNewestState` cleanup must restore the controller-requested extrinsic fixed/variable state, not unconditionally make `do_extrinsics` blocks variable. Never change `do_extrinsics_final_ba`; final BA keeps delay fixed.

- [ ] **Step 6: Wrap the existing realtime solve with validation**

Only when `finalizeTimeOffsetUpdate=true`, immediately before the existing `realtimeGraph_.optimise()` call capture `ParameterSnapshot`, cost and accepted scalar. Release/fix the scalar according to `beforeSolve()`, then solve normally. Build `TimeOffsetUpdateEvidence` from `summary().IsSolutionUsable()`, finite/bound checks, Ceres covariance, `allImuTimeOffsetEvaluationsValid()`, current visual support and post-solve cost. With the flag false, keep the pending/accepted scalar fixed and execute no calibration policy or diagnostics finalization.

Accept only when every design condition is true. On rejection:

```cpp
realtimeGraph_.restoreParameterSnapshot(snapshot);
realtimeGraph_.setTimeOffset(controller.lastAcceptedDelta());
realtimeGraph_.setTimeOffsetFixed(true);
realtimeGraph_.optimise(numIter, numThreads, verbose);
```

Validate the rerun solution before publishing. Do not call the public wrapper recursively. Only after acceptance/rerun may the existing landmark/state import into `fullGraph_` proceed. Add a loop-closure interleaving test with a pending `addStatesBacklog_`: reject the realtime candidate while full optimization is active, assert the full scalar and all existing full IMU factors retain the earlier accepted value, complete synchronization, then assert backlog factors are constructed with that same accepted value. A companion accepted case must apply the queued scalar before backlog construction.

- [ ] **Step 7: Run controller/backend tests**

Run `okvis_ceres_test`; expected: transition, extrinsic staging, accepted sync and full-state rollback tests pass. Also run `git diff --check` to catch accidental whitespace damage in the already-modified backend.

## Task 7: Make IMU Coverage and All Online Propagation Offset-Aware

**Files:**

- Create: `okvis_multisensor_processing/include/okvis/TimeOffsetTemporalCoverage.hpp`
- Create: `okvis_multisensor_processing/src/TimeOffsetTemporalCoverage.cpp`
- Modify: `okvis_multisensor_processing/include/okvis/ThreadedSlam.hpp`
- Modify: `okvis_multisensor_processing/src/ThreadedSlam.cpp`
- Create: `okvis_multisensor_processing/test/TestTimeOffsetTemporalCoverage.cpp`
- Modify: `okvis_multisensor_processing/CMakeLists.txt`

- [ ] **Step 1: Write failing overlap and no-extrapolation tests**

Add `TestTimeOffsetTemporalCoverage.cpp` to `okvis_multisensor_processing_test` before building. For measured IMU periods of 1 ms and 5 ms, assert:

```cpp
requiredOverlap = std::max(0.020, bound + 2.0 * medianImuPeriod);
```

Check both buffer ends against `[t0-bound-guard, t1+bound+guard]`. A candidate outside coverage is omitted; if accepted offset itself lacks coverage, the frame uses the last valid pose, freezes calibration and records `imuCoverageValid=false` rather than extrapolating.

- [ ] **Step 2: Replace the global fixed-overlap constant with a helper**

Implement and test these pure functions in `TimeOffsetTemporalCoverage.hpp/.cpp`:

```cpp
double medianImuPeriodSeconds(const ImuMeasurementDeque& measurements,
                              size_t maxIntervals = 100,
                              double fallbackSeconds = 0.005);
double requiredImuTemporalOverlapSeconds(double legacyOverlapSeconds,
                                         double residualBoundSeconds,
                                         double imuPeriodSeconds);
bool hasTimeOffsetCoverage(const ImuMeasurementDeque& measurements,
                           const Time& nominalStart,
                           const Time& nominalEnd,
                           double residualBoundSeconds,
                           double guardSeconds);
```

Then add private `ThreadedSlam` wrappers:

```cpp
double recentImuPeriodSeconds() const;
double imuTemporalOverlapSeconds() const;
bool propagateAtTimeOffset(double deltaSeconds,
                           kinematics::Transformation& T_WS,
                           SpeedAndBias& speedAndBias,
                           std::string* invalidReason) const;
```

The pure helper estimates period as the median of at most the latest 100 positive timestamp gaps and falls back to 5 ms until two samples exist. Add both new helper files to the `okvis_multisensor_processing` library. Use the dynamic overlap in initial wait, normal wait and old-IMU eviction. Keep at least one bracketing sample on both sides.

- [ ] **Step 3: Cache the accepted scalar across the optimization thread boundary**

Add `double acceptedTimeOffsetSeconds_ = 0.0;`. Update it only after joining `optimisationThread_`; detection-time propagation reads this cache and therefore does not access `estimator_` concurrently. The graph/controller remains authoritative. When the existing failed-initialization path calls `estimator_.clear()`, reset this cache to zero in the same branch.

- [ ] **Step 4: Route all propagation paths through the shifted helper**

Use `acceptedTimeOffsetSeconds_` for first-frame IMU selection/init, detection-time pose propagation and re-propagation after optimization. Use the current selected candidate only for current-frame state creation. Nominal `MultiFrame::timestamp()` is never changed.

- [ ] **Step 5: Run multisensor and existing tests**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_multisensor_processing_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_multisensor_processing_test$' --output-on-failure
```

Expected: coverage tests pass at both bound signs and old fixed-delay behavior passes with calibration disabled.

## Task 8: Add Read-Only Frontend Delay Hypotheses Before addStates/map Mutation

**Files:**

- Create: `okvis_frontend/include/okvis/TimeOffsetHypothesis.hpp`
- Create: `okvis_frontend/src/TimeOffsetHypothesis.cpp`
- Create: `okvis_frontend/test/TestTimeOffsetHypothesis.cpp`
- Modify: `okvis_frontend/include/okvis/Frontend.hpp`
- Modify: `okvis_frontend/src/Frontend.cpp`
- Modify: `okvis_frontend/CMakeLists.txt`
- Modify: `okvis_multisensor_processing/src/ThreadedSlam.cpp`

- [ ] **Step 1: Write failing deterministic score-order tests**

Add `TestTimeOffsetHypothesis.cpp` to `okvis_frontend_test` before building. Compare hypotheses lexicographically by: GP3P inliers, accepted initialized-map matches, contributing cameras, then lower median predicted reprojection error. A GP3P-not-computed candidate may still win on matches when every candidate has fewer than ten correspondences. Scores tied within one inlier/match and 0.25 px keep the accepted delta.

- [ ] **Step 2: Write a failing read-only integration test**

Build a small initialized synthetic map and multiframe. Snapshot estimator state count, landmark count, observation IDs, landmark IDs/qualities and every parameter value; call the scorer for three poses; assert all snapshots are identical afterward. Construct the center pose with a known projection shift and assert it is selected.

- [ ] **Step 3: Extract candidate collection from existing map matching**

Refactor the current `matchToMapByThread()` orchestration into a const helper that returns local landmark/keypoint candidates and `CameraMapMatchAccumulator` data without calling `addObservation`, `removeObservation`, `setLandmarkId`, cleanup or keyframe logic. The existing mutating map matcher calls this helper and then performs its current writes once, preserving all descriptor thresholds, 21.7 px spatial gate, EUCM projection and initialized/uninitialized handling.

- [ ] **Step 4: Separate pure GP3P model computation from outlier application**

Reuse the existing opengv non-central absolute-pose problem to compute model/inliers from local correspondences. The read-only path stops after model scoring. The existing primary/retry RANSAC path remains responsible for observation removal and pose assignment; it still executes exactly once for the selected hypothesis.

- [ ] **Step 5: Implement the public scorer and visual support result**

Add:

```cpp
TimeOffsetSelection scoreTimeOffsetHypotheses(
    const Estimator& estimator,
    const ViParameters& parameters,
    const MultiFramePtr& multiFrame,
    const std::vector<TimeOffsetPoseHypothesis>& hypotheses,
    double acceptedDeltaSeconds) const;

const TimeOffsetVisualSupport& lastTimeOffsetVisualSupport() const;
```

Only initialized landmarks enter hypothesis scoring. Require configured matches and contributing cameras before marking a score eligible. Save lightweight actual-path counts in `lastTimeOffsetVisualSupport_` regardless of diagnostics enablement when `do_time_offset` is true; reset it at each `dataAssociationAndInitialization()` entry.

- [ ] **Step 6: Insert selection before current-frame state creation**

After joining optimization and detecting features, but before `estimator_.addStates()`:

1. obtain `estimator_.timeOffsetHypotheses()`;
2. propagate each covered candidate from the last optimized nominal state;
3. call the read-only scorer against the previous initialized map;
4. call `estimator_.beginTimeOffsetCandidate(selection.deltaSeconds)`;
5. call the existing `addStates()` once, then existing `dataAssociationAndInitialization()` once;
6. pass `frontend_.lastTimeOffsetVisualSupport()` to the backend before realtime optimization.

The later `ThreadedSlam::optimisePublishMarginalise()` call passes `finalizeTimeOffsetUpdate=true`; all `optimiseRealtimeGraph()` calls made inside `Frontend.cpp` retain the default false.

If calibration is disabled, execute none of steps 1-4 and retain the old call order exactly. If scores are flat/insufficient, select the accepted scalar.

- [ ] **Step 7: Run frontend, multisensor and legacy decision tests**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_frontend_test okvis_multisensor_processing_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^(okvis_frontend_test|okvis_multisensor_processing_test)$' --output-on-failure
```

Expected: read-only snapshots are identical, known pose is selected, and disabled mode produces the same map IDs/observation decisions as the legacy path.

## Task 9: Add Bounded Time-Offset Diagnostics and Analysis Support

**Files:**

- Modify: `okvis_common/include/okvis/VioDiagnostics.hpp`
- Modify: `okvis_common/src/VioDiagnostics.cpp`
- Modify: `okvis_common/test/TestVioDiagnostics.cpp`
- Modify: `tools/accuracy_analysis/scripts/run_vio_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/test_run_vio_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py`
- Modify: `tools/accuracy_analysis/scripts/test_analyze_vio_causal_diagnostics.py`
- Modify: `okvis_multisensor_processing/src/ThreadedSlam.cpp`

- [ ] **Step 1: Write failing writer schema tests**

Add `vio_diag_time_offset.csv` and assert one fixed-width row per processed frame. The common-module record must use only standard/common types, not `TimeOffsetCalibrationState` or frontend structs, so `okvis_common` keeps its current dependency direction. It contains nominal/effective ns, nominal/candidate/accepted seconds, state/reason strings, posterior sigma, data information, required/available IMU ranges, coverage, BA release/accept/reject, extrinsics-fixed flag and at most `time_offset_hypothesis_count` scores.

```cpp
struct TimeOffsetDiagnosticRecord {
  uint64_t frameId = 0;
  uint64_t nominalTimestampNs = 0;
  std::optional<uint64_t> effectiveTimestampNs;
  double nominalDelaySeconds = 0.0;
  double candidateDeltaSeconds = 0.0;
  double acceptedDeltaSeconds = 0.0;
  std::string state = "WARMUP";
  std::string reason;
  std::optional<double> posteriorSigmaSeconds;
  std::optional<double> effectiveDataInformation;
  std::optional<uint64_t> requiredImuStartNs;
  std::optional<uint64_t> requiredImuEndNs;
  std::optional<uint64_t> availableImuStartNs;
  std::optional<uint64_t> availableImuEndNs;
  bool imuCoverageValid = false;
  bool scalarReleasedForBa = false;
  bool updateAccepted = false;
  bool extrinsicsFixed = true;
  struct Hypothesis {
    double deltaSeconds = 0.0;
    size_t gp3pInliers = 0;
    size_t acceptedInitialisedMatches = 0;
    size_t contributingCameras = 0;
    std::optional<double> medianPredictedReprojectionErrorPx;
    bool gp3pComputed = false;
    bool coverageValid = true;
  };
  std::vector<Hypothesis> hypotheses;
};
```

- [ ] **Step 2: Implement a separate bounded writer**

Add `writeTimeOffset(const TimeOffsetDiagnosticRecord&)`. Keep the CSV width fixed by serializing each hypothesis attribute as a semicolon-separated list column (`hypothesis_delta_s`, `hypothesis_gp3p_inliers`, `hypothesis_matches`, `hypothesis_cameras`, `hypothesis_median_reprojection_px`, `hypothesis_gp3p_computed`, `hypothesis_coverage_valid`). Read `time_offset_hypothesis_count` from the existing metadata map at `configure()` time and reject a record whose vector exceeds it. Increment diagnostics schema version to 2. Never add time-offset rows or columns to the landmark-event stream.

- [ ] **Step 3: Emit exactly once at the post-optimization safe point**

After acceptance or rollback completes and before publishing frame `k`, gather backend decision plus scorer results and write one record. `effectiveTimestampNs` uses the accepted delta for the value that will feed later frames; preserve nominal timestamp in the trajectory callback.

- [ ] **Step 4: Make the replay runner schema-aware**

For schema 1 outputs, retain current required-file behavior. For schema >=2 with metadata `do_time_offset=true`, require `vio_diag_time_offset.csv`, record all time-offset config values in `run_manifest.json`, and reject incomplete final rows. Do not invalidate already completed schema-1 experiments.

- [ ] **Step 5: Parse causal order in the analyzer**

Join time-offset rows to frame/RANSAC rows by `(timestamp_ns, frame_id)`. Summarize accepted delta, posterior sigma, state dwell time, bound contacts, rejection reasons and map support. Define causal-order events reproducibly:

```text
meaningful offset update:
  update_accepted == true and
  posterior_sigma is present and finite and
  abs(accepted_delta - previous_accepted_delta)
    >= max(0.0005 s, posterior_sigma)

visually unhealthy frame:
  primary GP3P attempted and failed, or
  accepted initialized-map matches < configured minimum, or
  contributing cameras < configured minimum

visual recovery:
  first of 3 consecutive healthy frames following at least one unhealthy frame

update precedes recovery:
  update timestamp < recovery timestamp <= update timestamp + 1.0 s
```

An accepted row with absent or non-finite posterior sigma is not a meaningful update. Report the number of eligible episodes, lead time and result per repeat. Label ordering evidence “replicated” only when the same classification appears in both repeats of a sequence; with fewer than two eligible repeats, label it insufficient rather than positive. Add synthetic joined-CSV fixtures for update-before-recovery, update-after-recovery, no meaningful update (including absent sigma) and no recovery. Do not claim a ground-truth delay from real sequences and do not load `vio_diag_landmark_events.csv` for this summary.

- [ ] **Step 6: Run C++ and Python diagnostics tests**

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc) --target okvis_common_test
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis -R '^okvis_common_test$' --output-on-failure
conda run -n okvis2x python -m unittest \
  tools.accuracy_analysis.scripts.test_run_vio_diagnostics \
  tools.accuracy_analysis.scripts.test_analyze_vio_causal_diagnostics
```

Expected: new schema passes, schema-1 compatibility passes, and analysis never opens the large landmark-event file for time-offset summaries.

## Task 10: Configure, Verify Numerics, and Run Causal Replay Acceptance

**Files:**

- Modify: `config/okvis2_eucm_EGO2.yaml`
- Modify: `okvis_ceres/test/TestImuTimeOffsetError.cpp`
- Modify after replay only: `workspace/ego2_results/202608_causal_diagnostics/analysis.md`

- [ ] **Step 1: Add disabled EGO2 configuration without changing nominal calibration**

Under `camera_parameters.online_calibration`, add the nine fields from Task 1 with `do_time_offset: false`. Keep this line byte-for-byte numerically unchanged:

```yaml
image_delay: 0.024869740
```

- [ ] **Step 2: Add synthetic known-offset recovery tests**

Generate IMU with angular acceleration, form fixed pose/speed states at boundaries shifted by a known `delta_d` in `{-0.010, -0.003, 0.004, 0.012}` seconds, release only the shared scalar and solve. Require every recovered offset within 1 ms. Repeat with static and constant-rate signals; require the estimate to stay within 0.1 ms of the zero prior and effective data information to remain zero within numeric tolerance.

- [ ] **Step 3: Verify disabled-path equivalence before enabling replay**

Run the full compiled suite:

```bash
cmake --build /home/chenguyuan/code/okvis_ws/build/okvis -j$(nproc)
ctest --test-dir /home/chenguyuan/code/okvis_ws/build/okvis --output-on-failure
git diff --check
```

Expected: all tests pass; no warnings/errors are introduced by new code; calibration-disabled unit/integration goldens match legacy behavior.

- [ ] **Step 4: Prepare a temporary enabled config**

Do not encode a sequence-specific delay. Create a replay-only config from the checked EGO2 config:

```bash
cp config/okvis2_eucm_EGO2.yaml /tmp/okvis2_eucm_EGO2_online_time_offset.yaml
sed -i 's/do_time_offset: false/do_time_offset: true/' /tmp/okvis2_eucm_EGO2_online_time_offset.yaml
```

Verify `image_delay` remains `0.024869740` and all residual candidates are centered on zero relative to that nominal value.

- [ ] **Step 5: Run repeated causal fixed and online replays over all 20260803-20260806 samples**

Use distinct, initially empty result roots:

```bash
conda run -n okvis2x python \
  tools/accuracy_analysis/scripts/run_vio_diagnostics.py \
  --reference-results-root workspace/ego2_results \
  --data-root /home/chenguyuan/data \
  --config config/okvis2_eucm_EGO2.yaml \
  --results-root workspace/ego2_results/202608_time_offset_calibration/fixed \
  --resume-all --repeats 2 --jobs 1

conda run -n okvis2x python \
  tools/accuracy_analysis/scripts/run_vio_diagnostics.py \
  --reference-results-root workspace/ego2_results \
  --data-root /home/chenguyuan/data \
  --config /tmp/okvis2_eucm_EGO2_online_time_offset.yaml \
  --results-root workspace/ego2_results/202608_time_offset_calibration/online \
  --resume-all --repeats 2 --jobs 1
```

These commands remain causal OKVIS2 replays. Slow execution is acceptable; do not enable future-frame or final-BA delay estimation.

- [ ] **Step 6: Analyze both roots with the corrected mocap policy**

Run the existing causal analyzer separately for fixed and online roots; it must use its 20260805 noon calibration correction rather than treating those three mocap streams as the default rigid-body calibration:

```bash
conda run -n okvis2x python \
  tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py \
  --diagnostics-root workspace/ego2_results/202608_time_offset_calibration/fixed \
  --data-root /home/chenguyuan/data \
  --output workspace/ego2_results/202608_time_offset_calibration/fixed_analysis

conda run -n okvis2x python \
  tools/accuracy_analysis/scripts/analyze_vio_causal_diagnostics.py \
  --diagnostics-root workspace/ego2_results/202608_time_offset_calibration/online \
  --data-root /home/chenguyuan/data \
  --output workspace/ego2_results/202608_time_offset_calibration/online_analysis
```

Compare APE RMSE/completion, predicted reprojection error, GP3P failure/inlier ratio, initialized map support, observation cleanup, landmark lifetime, accepted delay repeatability, sigma, bound hits and state transitions.

- [ ] **Step 7: Apply behavior-based acceptance gates**

Accept the implementation only if all conditions hold:

1. synthetic observable offsets recover within 1 ms;
2. static/constant-rate motion remains prior-dominated and does not drift;
3. all cameras share one scalar and both frontend/backend use the same sign/value;
4. rejected updates restore and rerun to an internally consistent graph;
5. high-angular sequences show repeatable improvement in old-map support, reprojection consistency, GP3P behavior, trajectory completion or APE;
6. normal sequences do not systematically regress; APE worsening over 10% requires improved completion or a documented reference-alignment limitation;
7. no accepted estimate repeatedly hits a bound; a bound hit is reported as a model/nominal-calibration warning;
8. under Task 9's event definitions, both repeats of each claimed sequence classify the meaningful accepted offset change before visual recovery; update-after-recovery or insufficient episodes are reported separately and cannot satisfy this gate.

- [ ] **Step 8: Record evidence without overstating causality**

Only after both replay sets complete, append a time-offset calibration section to `workspace/ego2_results/202608_causal_diagnostics/analysis.md`. Separate synthetic numerical proof, real-data causal timing evidence, negative/neutral sequences and unresolved residual effects. Do not report `39.25 ms` or any per-sequence offline optimum as ground truth.

## Final Self-Review Checklist

- [ ] Every design-spec section maps to a task: timestamps/configuration (1, 7), parameter ownership (2, 5, 6), shifted preintegration (3, 4), frontend feedback/hypotheses (7, 8), state machine/rollback/extrinsics (6), diagnostics (9), testing/acceptance (10).
- [ ] Run the no-placeholder scan required by `superpowers:writing-plans`; any match outside this checklist is a plan defect and must be replaced with an exact action.
- [ ] Confirm every later signature matches **Stable Contracts**, especially `delta_d` sign, fifth IMU block position and `TimeOffsetVisualSupport` spelling.
- [ ] Confirm `MultiFrame::timestamp()`, graph state timestamps, reprojection factor blocks, cleanup and loop-closure behavior are never rewritten.
- [ ] Confirm every IMU boundary path interpolates and refuses extrapolation at both residual bounds.
- [ ] Confirm full graph and final BA never release the scalar in v1.
- [ ] Confirm disabled mode skips hypothesis allocation/scoring and retains legacy decisions.
- [ ] Confirm diagnostics are bounded per frame and never expand the multi-gigabyte landmark-event CSV.
- [ ] Confirm no command in this plan stages or commits files.

## Completion Criteria

Implementation is complete only after Tasks 1-10 and the final checklist pass. Source changes remain uncommitted until the user explicitly requests a commit. A green build alone is insufficient: synthetic observability, rejection rollback, disabled equivalence and repeated causal EGO2 replay gates must all be satisfied.
