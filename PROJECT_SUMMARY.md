# 项目总结：机械臂避障规划与强化学习训练

> 基于 **PyBullet** 的六自由度 **Doosan H2017** 机械臂避障系统。
> 总体思路：先用 **RRT / LAPF-Informed RRT\*** 在关节空间离线规划全局路径库，再用 **强化学习（SAC/PPO）** 让机械臂沿随机采样的参考路径做**局部动态避障**。

---

## 1. 项目概述

| 项目 | 说明 |
|---|---|
| 仿真引擎 | PyBullet（240Hz 物理步长） |
| 机械臂 | Doosan H2017 六自由度，URDF + 胶囊体简化模型 |
| 全局规划 | RRT、LAPF 启发式 Informed RRT\* + B 样条平滑（离线路径库） |
| 局部避障 | SAC / PPO（stable_baselines3），Gymnasium 环境 |
| 感知 | RGB-D 相机（深度滤除法检测未知障碍物）+ 逐连杆胶囊距离 |
| 语言/环境 | Python 3.11，conda env `rl`，CUDA |

**核心分工：**
- **全局路径**：离线规划一批无碰撞关节空间轨迹，作为 RL 每回合随机采样的"参考路径"。
- **局部避障**：RL 策略沿参考路径前进，同时应对训练中随机出现的未知障碍物。

---

## 2. 系统架构

### 2.1 目录结构

```
arm_planning_changheng/
├── train_sac.py / train_ppo.py     # RL 训练入口（SAC / PPO）
├── test.py / test_1.py             # 加载训练好的模型执行避障并画轨迹
├── visualize.py                    # PyBullet 调试画点/画线工具
├── PROJECT_SUMMARY.md              # 本文件
│
├── arm_env/                        # ★ 仿真环境核心
│   ├── doosan_env.py               # Gymnasium 环境 LocalAvoidanceEnv
│   ├── robot.py                    # DoosanRobot：URDF、FK/IK、关节控制
│   ├── scene.py                    # SceneManager：地面、门框、动态障碍物
│   ├── reward.py                   # 奖励函数（引导式奖励塑形）
│   ├── visual_obstacle_detector.py # 相机深度差分 → 障碍物检测/重构
│   └── get_path.py
│
├── rrt_planning/                   # ★ 全局路径规划模块
│   ├── run_planning_simple.py      # 规划流程：RRT → LAPF-Informed RRT* → B样条
│   ├── load_json_run.py            # 从 path_joint.json 加载路径并执行
│   ├── rrtManipulator.py           # 基础 RRT 算法
│   ├── LAPF_informrrtstartManipulator_{1,2}_pis.py  # LAPF 启发式 Informed RRT*
│   ├── Bspline_2.py                # B 样条曲线平滑
│   └── module/LAPF.py              # 启发式路径生成器
│
├── util/
│   ├── camera.py                   # 相机 RGB/深度/掩码渲染，像素↔世界坐标换算
│   ├── geometry.py                 # 【新增】胶囊↔AABB 最近距离（纯 numpy）
│   └── utils.py                    # 碰撞检测、最短距离等
│
├── doosan/                         # 机械臂 URDF + 3D 网格资产（⚠️ 当前全为 0 字节，需恢复）
├── models/                         # 训练好的模型（ppo_checkpoint_*.zip 等）
└── logs/                           # TensorBoard 训练日志
```

### 2.2 核心流程

```
① 全局规划: 采样 M 组起止位形 → RRT/LAPF-RRT* → B样条 → 离线路径库
               (关节空间, 离线一次, 训练零开销)
② 局部避障: 每回合从路径库随机采样一条参考路径
               → LocalAvoidanceEnv → SAC/PPO 训练 → 加载模型实时避障
```

---

## 3. 全局路径规划（rrt_planning）

规划算法链路（[run_planning_simple.py](rrt_planning/run_planning_simple.py)）：

1. `RRT` 先生成一条初始可行路径（起步用）。
2. `LAPF_InformedRRTStar`（局部人工势场启发式 + Informed RRT\*）在 6 维关节空间规划无碰撞路径。
3. `Bspline_2.bspline_smooth_path()` 做 B 样条平滑，得到平滑关节角序列。

**定稿方案：离线路径库 + 任务随机化**

- 在满足**关节限位 + 静态环境无碰撞**约束下，均匀采样 M 组起止位形。
- 为每组起止位形规划一条无碰撞关节空间路径，B 样条平滑后**存入路径库**（每回合的参考路径）。
- 训练时每回合从库中均匀随机采样一条，作为该回合的任务定义（详见 §6 论文表述）。

> ⚠️ 当前仓库只加载 `path_joint.json`，**没有生成/保存路径库的脚本**，需按 §7 实施清单补充。

---

## 4. 局部避障强化学习环境（arm_env）

### 4.1 观测空间（定稿：42 维）

| 块 | 维度 | 说明 |
|---|---|---|
| q | 6 | 当前关节角 |
| q_v | 6 | 关节角速度（速度控制下反映真实限幅后的速度） |
| 局部路径误差 | 6 | 参考路径前瞻点（最近点+5）与 q 之差 |
| 目标关节误差 | 6 | 目标关节角与 q 之差 |
| 逐连杆距离（当前帧） | 6 | 每个连杆到最近障碍物距离 |
| 逐连杆距离（前 1 帧） | 6 | 历史堆叠 |
| 逐连杆距离（前 2 帧） | 6 | 历史堆叠 |

实现要点：
- 逐连杆距离替代了"危险向量 + 最短距离"；末端位置按需求移除。
- **K=3 帧历史堆叠**：每次相机检测时 push 新的距离向量进固定长度 deque，检测间隙沿用旧值（模拟真实离散传感器采样）。策略从时间序列隐式学习障碍物运动趋势，**无需显式跟踪滤波**。
- 奖励里的 `min_dist_to_obs` 取当前帧 `min(d_1..d_6)`，与观测分离。

### 4.2 动作空间（定稿：关节角速度 + 真实速度限幅）

```
action ∈ [-1,1]^6  （归一化）
target_vel = action × max_velocity × vel_scale     # 夹紧到 ±max_velocity
```

- `max_velocity` 即真实机械臂关节极限速度（URDF `jointMaxVelocity`，[robot.py:19](arm_env/robot.py#L19)）：
  ```python
  self.max_velocity = np.minimum(self.limit_velocity, np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.2]))
  ```
- `vel_scale`（建议 0.5~0.8）限制训练只用极限速度的一部分。
- `apply_velocity_control()`（`VELOCITY_CONTROL`）下发。
- **关节限位保护**：靠近关节上下限且速度朝外时将该关节速度清零。

### 4.3 感知方案（逐连杆距离，非上帝视角）

> 原则：**机器人自身连杆位置用正运动学（本体感知）；障碍物信息必须来自传感器。**

```
每个 step 的观测中，逐连杆距离 d_i 计算链路：

连杆胶囊体（自身模型，FK 求轴 + 半径常量）──┐
                                           ├─→ 胶囊↔包围盒 最近距离(纯numpy) → d_1..d_6
静态已知包围盒（场景创建时登记）────────────┘
相机检测的动态包围盒（depth差分→点云→AABB）─┘
```

- **静态已知障碍物**（桌子、门框、凸起）：`SceneManager` 创建时直接登记 AABB 参数（已知先验，无需感知）。
- **未知/动态障碍物**：`VisualObstacleDetector` 用深度差分（当前深度 vs 空场景基线）检测，点云拟合 AABB 重构，每 5 步刷新一次。检测出的盒参数只用于观测距离，**不加入碰撞集合**（避免双重碰撞）。
- **连杆胶囊体**：`get_link_capsules()` 用 `p.getLinkState`（FK）求相邻连杆世界系原点连线为轴线 + 每连杆半径常量。
- **距离计算**：`util/geometry.py`（新增）纯 numpy 实现 `capsule_to_aabb_dist()`，**不使用** `p.getClosestPoints`（上帝视角）。

> 相机视野限制是特性而非 bug：感知不到 FOV 外/被遮挡的障碍物，正是真实约束。
> 当前障碍物为"突然出现的静止障碍"（spawn 后不移动），深度差分天然匹配；历史堆叠已足以应对运动趋势。

### 4.4 碰撞检测（上帝视角真值）

```python
is_collision = ut.check_collision(self.robot.doosan_id,
                                  self.scene.obstacle_ids, self.scene.plane_id)
```

- 直接查 **PyBullet 接触点**（`p.getContactPoints`），检测机械臂 vs 所有真实障碍物（静态 + 动态）。
- 作为**终止条件 + 奖励惩罚**。部署时用不到，允许上帝视角。
- 与感知观测分离：**观测真实受限，终止判定用真值**。

### 4.5 奖励函数（arm_env/reward.py）

| 奖励分量 | 公式/行为 | 作用 |
|---|---|---|
| 目标吸引 r_goal | `400 × (prev_dist - curr_dist)` | 引导靠近目标 |
| 避障惩罚 r_obstacle | `< 0.08m 时 -20·exp(-5d)` | 接近障碍施压 |
| 趋势奖励 r_obs_trend | `gain × (curr_dist - prev_dist)` 裁剪 ±0.03 | 靠近罚/远离奖 |
| 脱离危险 r_escape | 距离从 <0.2 回到 >0.2 时 +5 | 鼓励脱困 |
| 平滑惩罚 r_smooth | `-0.05·‖Δaction‖²` | 抑制抖振 |
| 路径贴合 r_path | `-5·λ·min_dist_to_path` | 沿参考路径 |
| 时间惩罚 r_step | `-0.05·step/200` | 鼓励效率 |
| 稀疏终止 | 成功 +100 / 碰撞 -100 | 明确成功失败 |
| 汇总下限 | 截断到 ≥ -500 | 防梯度爆炸 |

---

## 5. 训练与测试

- **训练**：[train_sac.py](train_sac.py) / [train_ppo.py](train_ppo.py)
  - SAC（off-policy，replay buffer 1M、自动熵、SDE 探索噪声、CUDA），PPO 同理。
  - 并行 4 个环境（1 个 GUI + 3 个无渲染），`SubprocVecEnv`。
  - `EvalCallback`（每 5000 步）保最优模型到 `models/`；`CheckpointCallback`（每 20000 步）存 checkpoint。
  - 日志写 `logs/`，TensorBoard 可视化。
- **测试**：[test.py](test.py) / [test_1.py](test_1.py) 加载 `models/move/best_model.zip` 或 checkpoint，跑避障并画末端轨迹。
  - 观测改造后末端轨迹改用 `env.robot.get_hand_pos()`（不能再用 `obs[12:15]`）。

---

## 6. 论文表述：离线路径库 + 任务随机化

> 为提升局部策略的泛化能力、避免其对单一参考路径的过拟合，本文构建**离线全局路径库**：在满足关节限位与无碰撞约束下均匀采样 M 组起止位形，采用 LAPF 启发式 Informed RRT\* 为每组规划无碰撞关节空间路径，并经 B 样条平滑后入库。训练时每回合从库中均匀随机采样一条作为参考路径，构成**任务随机化**，使策略在多样化走廊上训练，收敛更稳。相比逐回合在线重规划，离线库将规划成本移出训练关键路径，避免采样规划器的随机性引入任务级噪声干扰学习；且全局路径仅需保证静态环境无碰撞，未知障碍由局部策略负责，符合层次化分工。

> To improve generalization and avoid overfitting to a single start–goal pair, we construct an offline library of global paths: M start–goal configurations are sampled under joint-limit and collision-free constraints, each planned by the LAPF-heuristic informed RRT\* and smoothed with B-spline curves. At the start of each episode, one reference path is uniformly sampled from the library, which amounts to task randomization and trains the policy over diverse corridors with stable convergence. Compared with per-episode online re-planning, the library removes planning cost from the training loop, avoids task-level noise from stochastic planners, and aligns with the hierarchical division of labor, where the global path only guarantees collision-freeness w.r.t. the static environment while unknown obstacles are handled by the local policy.

---

## 7. 实施清单（方案已定稿，待实现）

1. **新增 `util/geometry.py`**：胶囊↔AABB 最近距离（numpy，无 pybullet 查询）。
2. **[arm_env/robot.py](arm_env/robot.py)**：`get_link_capsules()`（FK + 每连杆半径常量）+ `apply_velocity_control()` 内部速度限幅。
3. **[arm_env/scene.py](arm_env/scene.py)**：静态障碍物 AABB 登记 + `get_obstacle_boxes()`。
4. **[arm_env/visual_obstacle_detector.py](arm_env/visual_obstacle_detector.py)**：`_create_aabb` 记录 `(center, half_extents)`，提供 `get_box_models()`。
5. **[arm_env/doosan_env.py](arm_env/doosan_env.py)**：
   - 打开检测器，reset 时采集空场景基线；
   - `step()` 每 5 步检测刷新障碍盒；
   - `get_obs()` 换成 q/q_v/local_err/q_err/逐连杆距离（K=3 堆叠），共 42 维；
   - 动作改为速度控制 + `max_velocity × vel_scale` 限幅 + 关节限位保护。
6. **全局路径库**：新增规划脚本，采样 M 组起止位形 → 规划 → B 样条平滑 → 保存为路径库；环境 reset 时随机采样一条。
7. **[test.py](test.py)**：末端轨迹改用 `env.robot.get_hand_pos()`。
8. **重新训练**：观测 42 维、动作语义变化，旧模型全部失效。

---

## 8. 已完成的改进

### 8.1 Windows → Linux 路径迁移 ✅

| 文件 | 原路径 | 改后 |
|---|---|---|
| [arm_env/robot.py:10-15](arm_env/robot.py#L10) | 硬编码 `/home/zww/...` + `D:/SoftwareInstallation/...` URDF | `os.path` 推导项目根 + 相对路径 `doosan/*.urdf` |
| [rrt_planning/load_json_run.py:42](rrt_planning/load_json_run.py#L42) | `D:\...\path_joint.json` | 项目根 + `rrt_planning/path_joint.json` |
| [rrt_planning/LAPF_informrrtstartManipulator_1_pis.py:613](rrt_planning/LAPF_informrrtstartManipulator_1_pis.py#L613) | `D:/.../doosan_planning/save_datas/raw_config2.txt` | 本项目 `rrt_planning/save_datas/`（自动建目录） |

### 8.2 环境依赖 ✅

- `pip install scikit-image`（装进 conda env `rl`）。
- ⚠️ `rl` 环境存在既有依赖冲突（torch 缺 `filelock/fsspec/jinja2`），与本次改动无关，但训练前需确认 torch 可用。

---

## 9. 已知问题与阻塞

| 问题 | 影响 | 处理 |
|---|---|---|
| 🔴 **`doosan/` 全部 76 个 URDF / 3D mesh 文件为 0 字节** | 机械臂无法加载，任何仿真无法运行 | **必须先从 Windows 原项目重新拷贝 `doosan/` 目录**（用 zip/压缩包传输，确保二进制完整） |
| 🟡 `rl` 环境 torch 缺依赖 | 训练可能失败 | 确认 `filelock/fsspec/jinja2` 或重装 torch |
| 🟡 观测/动作改造后旧模型失效 | 无法直接复用 | 改造完成后重新训练 |
| 🟡 仓库缺少路径库生成脚本 | 无法自动产生参考路径 | 按 §7 第 6 步补充 |

---

## 10. 运行方式

```bash
# 1. 恢复 doosan/ 资产（阻塞项）
# 2. 进入环境
conda activate rl

# 3. 生成全局路径库（待实现，目前只有单条 path_joint.json）
python rrt_planning/run_planning_simple.py

# 4. 训练
python train_sac.py      # 或 python train_ppo.py

# 5. 测试
python test.py
```

**依赖清单**：`pybullet`、`pybullet_utils`、`gymnasium`、`stable_baselines3`、`numpy`、`scipy`、`scikit-image`、`opencv-python`、`tensorboard`。
