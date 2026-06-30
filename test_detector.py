"""
集成测试: VisualObstacleDetector + LocalAvoidanceEnv

演示流程:
  1. 启动 env, 重置场景
  2. 采集空场景深度作为基线
  3. 在仿真中突然生成一个障碍物
  4. 用 VisualObstacleDetector 通过深度差分检测并重构
  5. 验证:
     - 重构的障碍物已加入 scene.obstacle_ids
     - get_obs() 中的 danger_vector / min_dist 能感知到它
     - check_collision 能检测到碰撞
     - RL 策略可以正常下发动作

运行:
  python test_detector.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pybullet as p
import pybullet_data
import numpy as np
import time

from arm_env.scene import SceneManager
from arm_env.robot import DoosanRobot
from arm_env.visual_obstacle_detector import VisualObstacleDetector
from util.camera import Camera
import util.utils as ut


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # 1. 初始化仿真组件
    robot = DoosanRobot()
    scene = SceneManager()
    scene.reset_scene()
    camera = Camera()

    # 把机械臂放到某个起始位置
    start_joints = [0, 0.3 * np.pi, 0.4 * np.pi, 0, 0.3 * np.pi, -0.5 * np.pi]
    robot.reset_joints(start_joints)

    # 2. 初始化视觉检测器 + 采集基线
    # 传入 robot.doosan_id, 检测时会自动从掩码中剔除机械臂自身
    detector = VisualObstacleDetector(
        camera, scene,
        exclude_body_ids=[robot.doosan_id],
    )
    baseline = camera.renderCameraDepthImage()
    detector.set_baseline(baseline)
    print("[INFO] 基线已采集 (空场景)")

    # 3. 空场景下验证: 不应检测到障碍物
    depth = camera.renderCameraDepthImage()
    obs_ids = detector.detect_and_reconstruct(depth)
    assert len(obs_ids) == 0, "空场景应检测不到障碍物"
    print("[INFO] 空场景验证通过: 无障碍物误报")

    # 4. 突然生成一个障碍物 (模拟实物中突然出现的物体)
    print("\n[INFO] 在场景中生成障碍物 (模拟实物)...")
    col = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.08]
    )
    vis = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.08, 0.08, 0.08],
        rgbaColor=[0, 1, 0, 1]
    )
    real_obs = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=[-0.2, -0.95, 1]
    )

    # 让物理引擎稳定
    for _ in range(30):
        p.stepSimulation()

    # 5. 视觉检测: 深度差分 → 点云 → OBB 重构
    print("[INFO] 执行视觉检测与重构...")
    current_depth = camera.renderCameraDepthImage()
    new_ids = detector.detect_and_reconstruct(current_depth)

    print(f"[INFO] 重构出 {len(new_ids)} 个障碍物, IDs: {new_ids}")
    print(f"[INFO] scene 中总障碍物数: {len(scene.obstacle_ids)}")

    # 6. 验证: 碰撞检测可以正常工作
    is_collision = ut.check_collision(
        robot.doosan_id, scene.obstacle_ids, scene.plane_id
    )
    print(f"[INFO] 当前碰撞状态: {'碰撞!' if is_collision else '安全'}")

    # 7. 验证: danger_features 能感知重构的障碍物
    min_dist, danger_vec = ut.get_obstacle_danger_features(
        robot.doosan_id, scene.obstacle_ids
    )
    print(f"[INFO] 距障碍物距离: {min_dist:.4f} m")
    print(f"[INFO] 危险方向向量: ({danger_vec[0]:.3f}, {danger_vec[1]:.3f}, {danger_vec[2]:.3f})")

    # 8. 手动下发动作用 RL 策略避障 (这里用随机噪声模拟)
    print("\n[INFO] 模拟 RL 策略下发动作 (向安全方向移动)...")
    q, _ = robot.get_joint_states()
    # 假设策略输出一个向上的动作
    escape_action = np.array([0.0, -0.05, 0.05, 0.0, 0.0, 0.0])
    target_joint = q + escape_action
    robot.apply_position_control(target_joint)

    for _ in range(50):
        p.stepSimulation()

    # 9. 验证避障后距离变化
    min_dist_after, danger_vec_after = ut.get_obstacle_danger_features(
        robot.doosan_id, scene.obstacle_ids
    )
    print(f"[INFO] 避障后距离: {min_dist_after:.4f} m (之前: {min_dist:.4f} m)")

    # 10. 每一帧都刷新检测 → 支持动态障碍物追踪
    print("\n[INFO] 启动实时检测循环 (按 Ctrl+C 退出)...")
    try:
        while p.isConnected():
            # 视觉检测 (可每 N 帧调用一次以节省性能)
            depth = camera.renderCameraDepthImage()
            detector.detect_and_reconstruct(depth)

            # 碰撞检测
            if ut.check_collision(robot.doosan_id, scene.obstacle_ids, scene.plane_id):
                print("[!] 碰撞!")

            p.stepSimulation()
            time.sleep(1 / 60)
    except KeyboardInterrupt:
        pass

    p.disconnect()


if __name__ == "__main__":
    main()
