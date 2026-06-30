import numpy as np
from util.utils import compute_obstacle_approach_reward

def compute_reward(
        current_step: int,
        curr_dist_to_goal: float,
        min_dist_to_obs: float,
        action: np.ndarray,
        prev_action: np.ndarray,
        min_dist_to_path: float,
        prev_dist_to_goal:float,
        prev_min_dist_to_obs:float,
        is_collision: bool,
        is_success: bool,
):
    reward = 0.0
    reward_info = {}

    # 目标吸引奖励
    r_attract  = 400.0 * (prev_dist_to_goal - curr_dist_to_goal)
    reward += r_attract
    reward_info['r_goal'] = float(r_attract)

    # 避障惩罚
    if min_dist_to_obs < 0.08:
        r_obstacle = -20 * np.exp(-5 * min_dist_to_obs)
    else:
        r_obstacle = 0.0
    reward += r_obstacle
    reward_info['r_obstacle'] = float(r_obstacle)

    # 离障碍是更近了还是更远了
    r_obs_trend = compute_obstacle_approach_reward(
        prev_min_dist=prev_min_dist_to_obs,
        curr_min_dist=min_dist_to_obs,
        gain=1.5,
        clip_value=0.03,
    )
    reward += r_obs_trend
    reward_info["r_obs_trend"] = r_obs_trend

    # 脱离危险奖励
    if prev_min_dist_to_obs < 0.2 and min_dist_to_obs > 0.2:
        r_escape = 5.0
    else:
        r_escape = 0.0
    reward += r_escape

    # 动作平滑惩罚
    alpha_smooth = 0.05
    r_smooth = -alpha_smooth * np.linalg.norm(action - prev_action)**2
    reward += r_smooth
    reward_info['r_smooth'] = float(r_smooth)

    # 初始路径贴合奖励
    lambda_path = np.clip((min_dist_to_obs - 0.1) / 0.2, 0.1, 5.0)
    r_path = -5.0 * lambda_path * min_dist_to_path
    reward += r_path
    reward_info['r_path'] = float(r_path)

    # 时间惩罚
    r_step = -0.05 * current_step / 200
    reward += r_step
    reward_info['r_step'] = float(r_step)

    # 稀疏奖励: 任务完成 与 碰撞惩罚
    r_terminal = 0.0
    if is_success:
        r_terminal = 100.0
    elif is_collision:
        r_terminal = -100.0

    reward += r_terminal
    reward_info['r_terminal'] = float(r_terminal)

    # 汇总奖励，并做下限截断防梯度爆炸
    reward = max(reward, -500.0)

    return reward, reward_info