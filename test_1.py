import time
import numpy as np
from arm_env.doosan_env import LocalAvoidanceEnv
from stable_baselines3 import PPO
import pybullet as p
from util import utils as ut

env = LocalAvoidanceEnv(gui=True)
env.render()
obs, info = env.reset()
phantom_obs_id = None # 用于视觉避障的虚拟障碍物ID
done = False
mode = "TRACKING"
rrt_path = env.smooth_reference_path
path_index = 0
model = PPO.load("models/stac/best_model.zip")
sim_step_counter = 0
obstacle_triggered = False
trigger_time = 50
time.sleep(1)
while not done:
    if path_index >= len(rrt_path):
        print("机械臂已经成功走完所有全局路点，顺利抵达终点！")
        break
    # min_dist_to_obs = env.get_vision_based_obstacle_dist()
    min_dist_to_obs, phantom_obs_id = ut.get_vision_based_obstacle_dist(
    env.camera,
    env.robot,
    env.baseline_depth,
    phantom_obs_id)
    # min_dist_to_obs = ut.get_dynamic_obstacle_dist(env.robot,env.scene.obstacle_ids)  # 障碍物距离
    q_current, _ = env.robot.get_joint_states()
    dist_to_path = np.linalg.norm(rrt_path[path_index:] - q_current, axis=1).min()

    if mode == "TRACKING":
        if min_dist_to_obs < 0.2 :
            print("突发障碍！切换为 RL 避障并重新规划回到路径！")
            mode = "RL_CONTROL"
            obs = env.get_obs()

    elif mode == "RL_CONTROL":
        if min_dist_to_obs > 0.2 and dist_to_path < 0.1:
            print("避障结束且已回到正轨！恢复 RRT 轨迹追踪。")
            path_index += np.argmin(np.linalg.norm(rrt_path[path_index:] - q_current, axis=1))
            mode = "TRACKING"

    # 执行控制
    if mode == "TRACKING":
        target_q = rrt_path[path_index]
        env.robot.apply_position_control(target_q)
        p.stepSimulation()
        sim_step_counter += 1
        current_error = np.linalg.norm(target_q - q_current)
        if current_error < 0.08:
            path_index += 1
        time.sleep(1 / 240.0)
        if not obstacle_triggered and sim_step_counter == trigger_time:
            current_hand_pos, _ = env.robot.get_hand_pos()
            env.scene._respawn_obstacle(current_hand_pos)
            p.performCollisionDetection()
            obstacle_triggered = True
            print("突发障碍物已生成！")

    elif mode == "RL_CONTROL":
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        sim_step_counter += env.control_skip
        time.sleep(0.02)

while p.isConnected():
    p.stepSimulation()