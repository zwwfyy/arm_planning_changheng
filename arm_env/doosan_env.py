import pybullet as p
import random
from pybullet_utils import bullet_client
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from pathlib import Path
from util.camera import Camera
import util.utils as ut
from arm_env.robot import DoosanRobot
from arm_env.scene import SceneManager
from arm_env.reward import compute_reward
from arm_env.visual_obstacle_detector import VisualObstacleDetector


class LocalAvoidanceEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, max_steps=400, gui=False, control_skip=20):
        super(LocalAvoidanceEnv, self).__init__()
        self.max_steps = max_steps
        self.control_skip = control_skip
        
        # 初始化 PyBullet 客户端
        if gui == False:
            self.p = bullet_client.BulletClient(connection_mode=p.DIRECT)
        else:
            self.p = bullet_client.BulletClient(connection_mode=p.GUI)

        p.setTimeStep(1.0 / 240.0)
        
        # 初始化机械臂、场景和相机
        self.robot = DoosanRobot()
        self.scene = SceneManager()
        self.camera = Camera()
        # self.detector = VisualObstacleDetector(self.camera, self.scene, exclude_body_ids=[self.robot.doosan_id])

        # 定义动作空间和观测空间
        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(31,), dtype=np.float32)

        # 其他环境参数
        self.success_dist_threshold = 0.1
        self.max_vel = 0.2
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_dist_to_goal = None
        self.prev_min_dist_to_obs = None
        self.baseline_depth = None
        self.global_path_json = Path(__file__).resolve().parents[1] / "rrt_planning" / "path_joint.json"

    # 从 JSON 文件加载全局路径
    def _load_global_path_from_json(self):
        with self.global_path_json.open("r", encoding="utf-8") as f:
            path = json.load(f)
        path = np.asarray(path, dtype=np.float32)
        if path.ndim != 2 or path.shape[1] != 6:
            raise ValueError(f"Global path JSON must be an N x 6 array, got {path.shape}")
        if np.nanmax(np.abs(path)) > 2 * np.pi:
            path = np.deg2rad(path)
        return path.astype(np.float32)

    # 重置环境
    def reset(self, seed=None, options=None):
        super(LocalAvoidanceEnv, self).reset(seed=seed)
        self.current_step = 0
        self.smooth_reference_path = self._load_global_path_from_json()
        if self.smooth_reference_path is None or len(self.smooth_reference_path) == 0:
            print("Warning: Global path is empty. Check the JSON file.")
        self.start_joints = self.smooth_reference_path[0].copy()
        self.goal_joints = self.smooth_reference_path[-1].copy()
        
        self.scene.reset_scene() # 重置场景
        self.robot.reset_joints(self.start_joints) # 重置机械臂到起始位置
        p.performCollisionDetection() # 确保碰撞检测状态更新
        
        # self.baseline_depth = self.camera.renderCameraDepthImage()
        # self.detector.set_baseline(self.baseline_depth)
        
        self.target_joints = np.array(self.goal_joints)
        self.obstacle_spawned = False
        self.trigger_step = random.randint(10, 120)
        # self.trigger_step = 9999
        
        self.prev_action = np.zeros(6, dtype=np.float32)
        self.prev_ema_reward = 0.0
        self.prev_dist_to_goal = self.get_dist_to_goal()
        obs = self.get_obs()
        self.prev_min_dist_to_obs = obs[-1]
        info = {}
        return obs, info
    
    # 与环境交互一步
    def step(self, action):
        self.current_step += 1
        
        q,_ = self.robot.get_joint_states()
        target_joint = np.array(q)+(np.array(action[0:6])/180*np.pi)
        self.robot.apply_position_control(target_joint)
        for _ in range(self.control_skip):
            p.stepSimulation()

        if not self.obstacle_spawned and self.current_step == self.trigger_step:
            self.scene._respawn_obstacle()
            p.performCollisionDetection()
            self.obstacle_spawned = True
            
        # 每隔5步使用相机视觉检测障碍物并重建环境
        # if self.current_step % 5 == 0:
        #     current_depth = self.camera.renderCameraDepthImage()
        #     self.detector.detect_and_reconstruct(current_depth)

        obs = self.get_obs()
        current_dist_goal = self.get_dist_to_goal()
        min_distance_obs = obs[-1]

        current_q = obs[:6]  
        distances_to_path = np.linalg.norm(self.smooth_reference_path - current_q, axis=1)
        min_dist_to_path = float(np.min(distances_to_path))


        is_collision = ut.check_collision(self.robot.doosan_id, self.scene.obstacle_ids, self.scene.plane_id)
        is_success = ut.check_success_joint_space(
            curr_dist_to_goal=current_dist_goal,
            dist_thresh=self.success_dist_threshold
        )

        # 计算奖励
        raw_reward, reward_info = compute_reward(
            current_step=self.current_step,
            curr_dist_to_goal=current_dist_goal,
            min_dist_to_obs=min_distance_obs,
            action=action,
            prev_action=self.prev_action,
            min_dist_to_path=min_dist_to_path,
            prev_dist_to_goal=self.prev_dist_to_goal,
            prev_min_dist_to_obs = self.prev_min_dist_to_obs,
            is_collision=is_collision,
            is_success=is_success,
        )


 
        self.prev_action = action.copy()
        self.prev_dist_to_goal = current_dist_goal

        terminated = is_success or is_collision
        truncated = self.current_step >= self.max_steps
        self.prev_min_dist_to_obs = min_distance_obs
        info = {
            "is_success": is_success,
            "is_collision": is_collision,
            "dist_to_goal": float(current_dist_goal),
            "min_dist_to_obs": float(min_distance_obs),
            **reward_info
        }

        if terminated or truncated:
            if info['is_collision']:
                print(f"Step {self.current_step}: collision, dist_to_goal={info['dist_to_goal']:.2f}")
            elif info['is_success']:
                print(f"Step {self.current_step}: success")
            else:
                print(f"Step {self.current_step}: timeout")

        return obs, raw_reward, terminated, truncated, info

    # 获取当前观测值
    def get_obs(self):
        q, q_v = self.robot.get_joint_states()
        pos,_ = self.robot.get_hand_pos()
        q_err = self.target_joints - q
        if hasattr(self, 'smooth_reference_path'):
            distances = np.linalg.norm(self.smooth_reference_path - q, axis=1)
            closest_idx = np.argmin(distances)
            target_idx = min(closest_idx + 5, len(self.smooth_reference_path) - 1)
            local_path_err = self.smooth_reference_path[target_idx] - q
        else:
            local_path_err = np.zeros(6, dtype=np.float32)

        min_dist, danger_vector = ut.get_obstacle_danger_features(self.robot.doosan_id, self.scene.obstacle_ids)
        obs = np.concatenate([q, q_v, pos, local_path_err, q_err, danger_vector * 10, [min_dist]]).astype(np.float32)
        return obs
    
    # 计算当前关节到目标位置的距离
    def get_dist_to_goal(self):
        q, _ = self.robot.get_joint_states()
        return np.linalg.norm(self.target_joints - q)

    # 配置相机视角
    def render(self):
        p.resetDebugVisualizerCamera(
            cameraDistance=3, cameraYaw=90, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

    def close(self):
        if p.isConnected():
            p.disconnect()

if __name__ == '__main__':
    env = LocalAvoidanceEnv(gui=True)
    obs, info = env.reset()
    env.camera.renderCameraRGBImage()
    env.camera.renderCameraDepthImage()
    env.camera.renderCameraMask()
    while p.isConnected():
        p.stepSimulation()

