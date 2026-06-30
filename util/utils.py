# 写碰撞，距离计算等
import pybullet as p
import numpy as np

# 计算全身避障特征：返回全局最短距离(1维) 和相对危险向量(3维)
def get_obstacle_danger_features(robot_id, obstacle_ids, safe_dist=0.5, max_vec_norm=0.5):
    if not obstacle_ids:
        return float(safe_dist), np.array([0.0, 0.0, safe_dist], dtype=np.float32)
    global_min_dist = float('inf')
    best_danger_vector = np.array([0.0, 0.0, safe_dist], dtype=np.float32)
    for obs_id in obstacle_ids:
        closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=safe_dist)
        if not closest_points:
            continue
        for pt in closest_points:
            if pt[8] < global_min_dist:
                global_min_dist = pt[8]
                closest_pt_on_robot = np.array(pt[5])
                closest_pt_on_obs = np.array(pt[6])
                best_danger_vector = closest_pt_on_obs - closest_pt_on_robot
    if global_min_dist == float('inf'):
        return float(safe_dist), np.array([0.0, 0.0, safe_dist], dtype=np.float32)
    norm = np.linalg.norm(best_danger_vector)
    if norm > max_vec_norm and norm > 1e-8:
        best_danger_vector = best_danger_vector / norm * max_vec_norm
    return float(global_min_dist), best_danger_vector.astype(np.float32)

# 机械臂与挨个障碍物的碰撞
def check_collision(robot_id, obstacle_ids, plane_id):
    for obs_id in obstacle_ids:
        obs_contacts = p.getContactPoints(bodyA=robot_id, bodyB=obs_id)
        if len(obs_contacts) > 0:
            return True
    plane_contacts = p.getContactPoints(robot_id, plane_id) # 检查与地面的碰撞，但要排除底座
    for contact in plane_contacts:
        link_index = contact[3]
        if link_index > 1:
            return True
    return False

# 机械臂与所有的碰撞
def check_all_collision(node, robot):
    for i in range(robot.DOF):
        p.resetJointState(robot.doosan_id, i, node[i])
    p.performCollisionDetection()
    if len(p.getContactPoints(robot.doosan_id)) == 0:
        return False
    else:
        return True

# 计算成功距离
def check_success_joint_space(curr_dist_to_goal,dist_thresh=0.08):
    return bool(curr_dist_to_goal < dist_thresh)

# 若离障碍更近，则给负奖励；更远则给正奖励。
def compute_obstacle_approach_reward(prev_min_dist, curr_min_dist, gain=2.0, clip_value=0.05):
    if prev_min_dist is None:
        return 0.0

    delta = np.clip(curr_min_dist - prev_min_dist, -clip_value, clip_value)
    return gain * delta

# 计算机械臂与突发障碍物之间的最短距离
def get_dynamic_obstacle_dist(robot, obs_ids):
    if len(obs_ids) == 1:
        return 10.0
    if len(obs_ids) > 0:
        dynamic_obs_id = obs_ids[-1]
        closest_points = p.getClosestPoints(
            bodyA=robot.doosan_id,
            bodyB=dynamic_obs_id,
            distance=1.0 
        )
        if closest_points:
            return float(min([pt[8] for pt in closest_points]))
    return 1.0


