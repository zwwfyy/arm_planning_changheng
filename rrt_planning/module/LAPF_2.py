import math
import numpy as np
import pybullet as p
import matplotlib

matplotlib.use('TkAgg')
from arm_env.robot import DoosanRobot
from visualize import Visualize

class HeuristicPathGenerator:
    def __init__(self, robot, x_start, x_goal, step_len=0.2, N_rays=3,
                 tau_free=8, eta=1.5):
        """
        robot: 机器人对象
        x_start: 起点关节坐标
        x_goal: 终点关节坐标
        step_len: 关节空间步长
        N_rays: 采样方向数量
        tau_free: 自由移动阈值
        eta: 目标重定向系数
        """
        # 起点和终点节点
        self.s_start = np.array(x_start)
        self.s_goal = np.array(x_goal)
        self.robot = robot

        # 路径规划参数
        self.step_len = step_len
        self.N_rays = N_rays
        self.tau_free = tau_free
        self.eta = eta

        # 获取关节限制
        self.joint_lower = np.array(robot.limit_lower)
        self.joint_upper = np.array(robot.limit_upper)
        self.dim = len(x_start)  # 关节维度

        self.vis = Visualize()

    def generate(self, max_iters=500, stuck_limit=5, wall_follow_steps=3,
                 jitter_max_attempts=6, jitter_max_radius=0.2):
        """
        生成从起点到终点的关节空间路径

        参数:
        max_iters: 最大迭代次数
        stuck_limit: 卡住检测阈值
        wall_follow_steps: 沿墙行走步数
        jitter_max_attempts: 抖动最大尝试次数
        jitter_max_radius: 抖动最大半径
        """
        # 计算初始方向（关节空间）
        d_start = self.get_joint_dir(self.s_start, self.s_goal)
        # 生成均匀分布的关节空间方向
        D_start = [d_start] + self.uniform_joint_directions(self.N_rays)
        num = len(D_start)
        all_paths = []  # 存储所有找到的路径

        # 对每个初始方向进行路径搜索
        for idx, init_dir in enumerate(D_start):
            print(f"尝试方向 {idx + 1}/{num}")
            path = [self.s_start]
            q_cur = self.s_start
            cur_dir = init_dir
            reflect_free = 0
            goal_free = 0
            stuck_counter = 0
            last_dist = self.joint_dist(q_cur, self.s_goal)
            it = 0
            while self.joint_dist(q_cur, self.s_goal) > self.step_len * 0.5 and it < max_iters:

                # 1. 融合势场方向和当前方向
                pf_dir = self.potential_field_direction(q_cur, self.s_goal)
                fused = cur_dir + pf_dir
                fused_norm = np.linalg.norm(fused)
                if fused_norm > 1e-8:
                    fused = fused / fused_norm
                else:
                    fused = self.get_joint_dir(q_cur, self.s_goal)

                # 2. 尝试移动
                q_next = q_cur + fused * self.step_len  # 减小步长
                q_next = np.clip(q_next, self.joint_lower, self.joint_upper)

                # 下一个移动点是否在障碍物里
                if self.check_collision(q_next):
                    q_next = q_cur

                # 3. 连接段碰撞检测
                if not self.check_path_collision(q_cur, q_next):
                    q_cur = q_next
                    path.append(q_cur)
                    reflect_free += 1
                    goal_free += 1
                    # 检查是否接近目标
                    if self.joint_dist(q_cur, self.s_goal) < self.step_len:
                        break
                else:
                    # 碰撞处理
                    q_hit = self.critical_joint_point(q_cur, q_next)
                    q_cur = q_hit
                    path.append(q_cur)
                    reflect_free = goal_free = 0

                    # 计算障碍物法线方向
                    normal = self.obstacle_joint_normal(q_cur)
                    tangent = self.get_joint_tangent(normal)

                    # 尝试沿墙行走
                    succeeded = False
                    for sign in [1, -1]:
                        tdir = tangent * sign
                        q_try = q_cur
                        collision = False
                        for step in range(wall_follow_steps):
                            q_try = q_try + tdir * self.step_len * 0.5
                            q_try = np.clip(q_try, self.joint_lower, self.joint_upper)
                            if self.check_path_collision(q_cur, q_try):
                                collision = True
                                break
                        if not collision:
                            q_cur = q_try
                            path.append(q_cur)
                            succeeded = True
                            break

                    if not succeeded:
                        cur_dir = self.reflect(cur_dir, normal)

                # 定期重定向
                if reflect_free >= self.tau_free:
                    cur_dir = self.get_joint_dir(q_cur, self.s_goal)
                    reflect_free = 0

                if goal_free >= self.eta * self.tau_free:
                    t_dir = self.get_joint_dir(q_cur, self.s_goal)
                    half = cur_dir + t_dir
                    half_norm = np.linalg.norm(half)
                    if half_norm > 1e-8:
                        half = half / half_norm
                    q_bis = q_cur + half * self.step_len * 0.3
                    q_bis = np.clip(q_bis, self.joint_lower, self.joint_upper)
                    if not self.check_path_collision(q_cur, q_bis):
                        q_cur = q_bis
                        path.append(q_cur)
                    goal_free = 0

                # 卡住检测
                cur_dist = self.joint_dist(q_cur, self.s_goal)
                if cur_dist < last_dist - 1e-3:
                    stuck_counter = 0
                else:
                    stuck_counter += 1
                last_dist = cur_dist

                # 卡住处理
                if stuck_counter >= stuck_limit:
                    success = False
                    for attempt in range(jitter_max_attempts):
                        r = (attempt + 1) / jitter_max_attempts * jitter_max_radius
                        jitter = self._sample_joint_jitter(radius=r)
                        q_try = q_cur + jitter
                        q_try = np.clip(q_try, self.joint_lower, self.joint_upper)
                        if not self.check_path_collision(q_cur, q_try):
                            q_cur = q_try
                            path.append(q_cur.copy())
                            stuck_counter = 0
                            success = True
                            break
                    if not success:
                        stuck_counter = 0

            # 检查最终路径
            final_dist = self.joint_dist(q_cur, self.s_goal)
            if final_dist <= self.step_len * 2.0:
                if not self.check_path_collision(q_cur, self.s_goal):
                    path.append(self.s_goal.copy())
                    all_paths.append(path)

        # 选择最优路径
        if all_paths:
            all_paths.sort(key=lambda p: self.evaluate_path_quality(p))
            best_path = all_paths[0]
            print(f"初始路径生成成功! 路径点数: {len(best_path)}")
            return best_path
        else:
            print("⚠ 未找到可行路径，返回直线")
            return [self.s_start.copy(), self.s_goal.copy()]

    def evaluate_path_quality(self, path):
        """评估路径质量"""
        if len(path) < 2:
            return float('inf')

        # 路径长度
        length_cost = sum(self.joint_dist(path[i], path[i + 1]) for i in range(len(path) - 1))

        # 路径光滑度
        smoothness_cost = 0
        for i in range(1, len(path) - 1):
            vec1 = path[i] - path[i - 1]
            vec2 = path[i + 1] - path[i]
            if np.linalg.norm(vec1) > 1e-8 and np.linalg.norm(vec2) > 1e-8:
                cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                smoothness_cost += 1 - cos_angle

        return length_cost + smoothness_cost * 0.05

    # 关节空间辅助函数
    def get_joint_dir(self, A, B):
        """获取关节空间方向向量"""
        direction = B - A
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            return direction / norm
        else:
            return np.random.uniform(-1, 1, self.dim)

    def uniform_joint_directions(self, N):
        """生成关节空间的均匀方向"""
        directions = []
        for _ in range(N):
            direction = np.random.normal(0, 1, self.dim)
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction = direction / norm
            directions.append(direction)
        return directions

    def joint_dist(self, A, B):
        """计算关节空间距离"""
        return np.linalg.norm(A - B)

    def check_collision(self, joint_config):
        """检查单个关节配置是否碰撞"""
        try:
            # 设置机械臂关节状态
            for i in range(self.robot.DOF):
                p.resetJointState(self.robot.robotID, i, joint_config[i])

            # 执行碰撞检测
            p.performCollisionDetection()
            contact_points = p.getContactPoints(robot.robotID)
            if len(contact_points) > 0:
                    return True
            else:
                    return False

        except Exception as e:
            print(f"碰撞检测错误: {e}")

    def check_path_collision(self, A, B):
        """检查关节空间路径段是否碰撞"""
        steps = max(2, int(self.joint_dist(A, B) / self.step_len * 3))
        for t in np.linspace(0, 1, steps):
            point = A + t * (B - A)
            if self.check_collision(point):
                return True
        return False

    def critical_joint_point(self, A, B):
        """找到关节空间路径段的第一个碰撞点"""
        low, high = 0.0, 1.0
        for _ in range(6):
            mid = (low + high) / 2
            point = A + mid * (B - A)
            if self.check_collision(point):
                high = mid
            else:
                low = mid
        return A + low * (B - A)

    def obstacle_joint_normal(self, p):
        """计算关节空间障碍物法线方向"""
        normal = np.zeros(self.dim)
        # 基于关节限制
        for i in range(self.dim):
            if p[i] - self.joint_lower[i] < 0.2:
                normal[i] = 1.0
            elif self.joint_upper[i] - p[i] < 0.2:
                normal[i] = -1.0
        # 数值梯度法
        if np.linalg.norm(normal) < 0.5:
            epsilon = 0.01
            base_collision = self.check_collision(p)
            for i in range(self.dim):
                test_p = p.copy()
                test_p[i] += epsilon
                if self.check_collision(test_p) != base_collision:
                    normal[i] = -1.0 if base_collision else 1.0
        norm = np.linalg.norm(normal)
        if norm > 1e-8:
            return normal / norm
        else:
            return np.array([1.0 if i == 0 else 0.0 for i in range(self.dim)])

    def get_joint_tangent(self, normal):
        # 选择与法线正交的基向量
        tangent = np.ones(self.dim) - normal * np.dot(normal, np.ones(self.dim))
        norm = np.linalg.norm(tangent)
        if norm > 1e-8:
            return tangent / norm
        else:
            # 如果失败，使用随机方向
            tangent = np.random.uniform(-1, 1, self.dim)
            tangent = tangent - np.dot(tangent, normal) * normal
            norm = np.linalg.norm(tangent)
            if norm > 1e-8:
                return tangent / norm
            return np.array([0.0 if i == 0 else 1.0 for i in range(self.dim)])

    def reflect(self, d, n):
        """反射方向计算"""
        dot = np.dot(d, n)
        return d - 2 * dot * n

    def potential_field_direction(self, p_node, q_tar):
        """使用真实接触信息的势场方向"""
        # 吸引力
        to_goal = q_tar - p_node
        d_goal = np.linalg.norm(to_goal)
        if d_goal > 1e-8:
            attr = (to_goal / d_goal) * 1.0
        else:
            attr = np.zeros(self.dim)

        # 排斥力（基于真实接触信息）
        rep = np.zeros(self.dim)

        try:
            # 获取接触点信息
            for i in range(self.robot.DOF):
                p.resetJointState(self.robot.robotID, i, p_node[i])

            p.performCollisionDetection()
            contact_points = p.getContactPoints(self.robot.robotID)
            if contact_points:
                # 使用接触法向量计算排斥方向
                obstacle_normal = self.obstacle_joint_normal(p_node)
                if np.linalg.norm(obstacle_normal) > 1e-8:
                    # 基于最近接触距离计算排斥力强度
                    min_distance = min(contact[8] for contact in contact_points)
                    rep_strength = self.calculate_repulsion_strength(min_distance)
                    rep = obstacle_normal * rep_strength

        except Exception as e:
            print(f"势场计算错误: {e}")

        # 合力
        total = attr + rep*0.3 # 调整权重

        norm = np.linalg.norm(total)
        if norm > 1e-8:
            return total / norm
        else:
            return self.get_joint_dir(p_node, q_tar)

    def calculate_repulsion_strength(self, distance):
        """根据距离计算排斥力强度"""
        if distance >= 0.1:  # 10cm以外，排斥力很小
            return 0.0
        elif distance >= 0:  # 0-10cm，线性增加
            return (0.1 - distance) * 10
        else:  # 穿透，强排斥
            return min(abs(distance) * 20, 2.0)

    def _sample_joint_jitter(self, radius):
        """生成关节空间随机抖动"""
        direction = np.random.normal(0, 1, self.dim)
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction = direction / norm
        r = np.random.uniform(0, radius)
        return direction * r

    def visualize_path(self, path):
        """可视化路径"""
        if len(path) < 2:
            print("路径太短，无法可视化")
            return
        # 将关节路径转换为笛卡尔空间路径
        cartesian_path = []
        for joint_point in path:
            pos, orn = self.robot.kinematics(joint_point)
            cartesian_path.append(pos)
        self.vis.plot_path(cartesian_path,[0.1, 0.1, 0.1], [0.8, 0, 0.8], 4, 2)