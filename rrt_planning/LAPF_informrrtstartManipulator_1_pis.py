# 修改记录：在_1的基础上增加偏向采样
import numpy as np
from scipy.spatial.transform import Rotation as R
from visualize import Visualize
import time
import math
from util.utils import check_all_collision
from module.LAPF import HeuristicPathGenerator
import Bspline_2


class LAPF_InformedRRTStar:
    def __init__(self, start, goal, step_delta, iter_max, sample_rate, search_radius, robot, use_heuristic=True):
        # 初始化配置参数
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_delta = float(step_delta)
        self.iter_max = int(iter_max)
        self.sample_rate = float(sample_rate)
        self.robot = robot
        self.search_radius = float(search_radius)
        self.use_heuristic = use_heuristic

        # RRT数据结构
        self.rrt_value = [self.start]
        self.rrt_parent = [-1]
        pos, orn = self.robot.kinematics(start)
        self.rrt_3Dposition = [pos]
        self.rrt_3Dorn = [orn]
        self.rrt_cost = [0.0]  # 从起点到各节点的累积代价

        # Informed RRT*专用参数
        self.best_path = None
        self.best_cost = float('inf')
        self.c_min = np.linalg.norm(self.goal - self.start)
        self.dim = len(start)

        # 标记是否找到初始路径
        self.found_initial_path = False
        self.initial_path_iterations = 0  # 用于统计找到初始路径的迭代次数

        # 椭圆采样参数
        self.x_center = (self.start + self.goal) / 2
        self.C = np.eye(self.dim)  # 初始化为单位矩阵
        self.a = self.c_min / 2
        self.b = 1.0  # 初始短轴长度

        # 性能统计
        self.planning_time = 0
        self.final_path_length = 0

        # 启发式路径引导参数
        self.heuristic_path = None
        self.heuristic_path_index = 0
        self.last_guided_sample = None
        self.min_sampling_radius = 0.05  # 最小采样半径
        self.guided_sampling_prob = 0.8  # 在寻找初始路径阶段使用引导采样的概率
        self.initial_path_bias = 0.9  # 在初始路径搜索阶段偏向目标的概率

    def generate_heuristic_path(self):
        """生成启发式路径用于引导采样"""
        print("=== 生成启发式引导路径 ===")

        # 创建启发式路径生成器
        heuristic_planner = HeuristicPathGenerator(
            robot=self.robot,
            x_start=self.start,
            x_goal=self.goal,
        )
        # 生成启发式路径
        path = heuristic_planner.generate()
        if path and len(path) > 1:
            print(f"启发式路径生成成功，包含 {len(path)} 个点")
            self.heuristic_path = [np.array(point) for point in path]
            self.heuristic_path_index = 0
            self.last_guided_sample = None
            return True
        else:
            print("警告: 启发式路径生成失败，将使用标准RRT*")
            self.heuristic_path = None
            return False

    def get_min_obstacle_distance(self, point):
        """估算点到障碍物的最小距离"""
        max_distance = 0.5  # 最大搜索距离
        step_size = 0.05

        min_distance = max_distance
        # 在几个关键方向上检查
        for distance in np.arange(step_size, max_distance, step_size):
            # 随机扰动方向
            direction = np.random.randn(self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-6)

            test_point = point + direction * distance
            test_point = np.clip(test_point, self.robot.limit_lower, self.robot.limit_upper)

            if check_all_collision(test_point, self.robot):
                min_distance = min(min_distance, distance)

        return min_distance

    def step_forward_along_path(self):
        """
        沿着启发式路径周围采样
        """
        if not self.heuristic_path or self.heuristic_path_index >= len(self.heuristic_path) - 1:
            # 如果已经到达路径末端，回退到目标偏向采样
            if np.random.random() < self.initial_path_bias:
                return self.goal

        # 前进到下一个路径点
        next_idx = min(self.heuristic_path_index + 3, len(self.heuristic_path) - 1)
        q_ref = self.heuristic_path[next_idx]

        # 计算采样半径
        r_max = self.get_min_obstacle_distance(q_ref)

        if self.last_guided_sample is not None:
            dist_to_last = np.linalg.norm(q_ref - self.last_guided_sample)
            r = min(r_max, max(dist_to_last * 0.5, self.step_delta))
        else:
            r = min(r_max, self.step_delta * 3)

        # 确保采样半径不小于最小值
        r = max(r, self.min_sampling_radius)

        # 在参考点周围球形采样
        attempt_count = 0
        while attempt_count < 10:
            # 在高维单位球内采样
            direction = np.random.randn(self.dim)
            direction = direction / (np.linalg.norm(direction) + 1e-6)
            # 随机半径（偏向较小半径以保持路径连续性）
            r_sampled = np.random.uniform(0, r)
            # 计算新点
            new_point = q_ref + direction * r_sampled
            # 边界检查
            new_point = np.clip(new_point, self.robot.limit_lower, self.robot.limit_upper)
            # 确保不是无效点
            if np.any(np.isnan(new_point)):
                attempt_count += 1
                continue
            # 更新记录
            self.last_guided_sample = new_point.copy()
            self.heuristic_path_index = next_idx
            return new_point
        # 如果多次尝试失败，返回参考点本身
        self.heuristic_path_index = next_idx
        return q_ref.copy()

    def initial_path_sample(self):
        """初始路径搜索阶段的采样策略"""
        if self.heuristic_path is not None and np.random.random() < self.guided_sampling_prob:
            return self.step_forward_along_path()
        # 以一定概率偏向目标
        elif np.random.random() < self.initial_path_bias:
            return self.goal
        else:
            return self.sample_within_bounds()

    def optimize_phase_sample(self):
        """优化阶段的采样策略"""
        if self.best_cost < float('inf'):
            # 90%概率使用椭圆采样，10%概率随机采样
            if np.random.random() < 1:
                return self.informed_sample()
        return self.random_sample()

    def planning(self, plot=False):
        # 生成启发式路径
        if self.use_heuristic:
            heuristic_success = self.generate_heuristic_path()
            if not heuristic_success:
                self.heuristic_path = None
        print("开始路径规划...")
        start_time = time.time()
        # 分阶段规划：先找初始路径，然后优化
        for i in range(self.iter_max):
            if self.found_initial_path:
                path,cost = self.extract_solution()
                if i % 100 == 0:
                    status = "优化阶段" if self.found_initial_path else "初始路径搜索"
                    print(f"迭代进度: {i}/{self.iter_max}, 阶段: {status}, 当前最佳代价: {cost*100:.3f}")

            # 根据当前阶段选择采样策略
            if not self.found_initial_path:
                node_rand = self.initial_path_sample()
            else:
                node_rand = self.optimize_phase_sample()

            # 扩展树
            node_near, index_near = self.nearest_node(node_rand)
            node_new = self.new_node(node_near, node_rand)

            if not self.is_collision(node_near, node_new):
                # 寻找邻近节点
                neighbor_indices = self.find_neighbors(node_new)

                # 选择最优父节点,并重连接
                best_parent, rewire_count = self.choose_parent_and_rewire_cached(node_new, neighbor_indices)

                if best_parent is not None:
                    # 检查是否到达目标
                    if self.reach_goal(node_new):
                        path, cost = self.extract_solution()
                        if not self.found_initial_path:
                            time1 = time.time()
                            init_time = time1-start_time
                            self.found_initial_path = True
                            self.initial_path_iterations = i
                            self.best_cost = cost
                            self.best_path = path
                            self.update_ellipse(cost)  # 初始化椭圆参数
                            print(f"\n=== 找到初始路径（迭代 {i}）===")
                            print(f"初始路径代价: {cost*100:.3f}")
                            print(f"找到初始路径时间为：{init_time:.3f}")
                            print("切换到优化阶段...")
                        elif cost < self.best_cost:
                            # 找到更优路径
                            self.update_ellipse(cost)
                            old_cost = self.best_cost
                            self.best_cost = cost
                            self.best_path = path
                            print(f"迭代 {i}: 找到更优路径，代价: {cost*100:.3f} (改进: {old_cost - cost:.3f})")
        # 最终结果处理
        self.planning_time = time.time() - start_time
        self.best_path, cost = self.extract_solution()
        if self.best_path:
            if plot:
                self.visualize()
            # 路径剪枝
            # self.best_path = self.path_pruning_3d(self.best_path)
            self.final_path_length = cost
            print(f"\n=== 规划结果 ===")
            print(f"规划时间: {self.planning_time:.3f}s")
            print(f"找到初始路径的迭代次数: {self.initial_path_iterations}")
            print(f"最终路径长度: {self.final_path_length * 100:.3f}cm")
            if self.heuristic_path:
                print(f"启发式路径引导: 启用")

            return self.best_path
        else:
            print("规划失败!")
            return None

    # 以下方法保持不变...
    def path_pruning_3d(self, path_tuple):
        """3D路径剪枝：跳过中间节点，直接连接可见节点"""
        config_list, cartesian_list, euler_list = path_tuple

        if len(cartesian_list) <= 2:
            return path_tuple

        # 剪枝算法
        pruned_indices = [0]  # 起点索引
        current_index = 0

        while current_index < len(cartesian_list) - 1:
            # 从当前节点开始，寻找最远可以直连的节点
            farthest_valid_index = current_index + 1

            for test_index in range(current_index + 2, len(cartesian_list)):
                # 检查两个配置点之间是否有碰撞
                if not self.check_collision_between_configs(
                        config_list[current_index],
                        config_list[test_index]
                ):
                    farthest_valid_index = test_index

            pruned_indices.append(farthest_valid_index)
            current_index = farthest_valid_index

        # 提取剪枝后的路径
        pruned_config = [config_list[i] for i in pruned_indices]
        pruned_cartesian = [cartesian_list[i] for i in pruned_indices]
        pruned_euler = [euler_list[i] for i in pruned_indices]

        return pruned_config, pruned_cartesian, pruned_euler

    def check_collision_between_configs(self, config1, config2):
        """检查两个关节配置之间的路径是否有碰撞"""
        num_samples = max(2, int(np.linalg.norm(np.array(config2) - np.array(config1)) / self.step_delta * 10))
        samples = np.linspace(config1, config2, num_samples)

        for node in samples:
            if check_all_collision(node, self.robot):
                return True

        return False

    def choose_parent_and_rewire_cached(self, node_new, neighbor_indices):
        """使用碰撞检测缓存选择父节点并进行重连接"""
        if not neighbor_indices:
            return None, 0

        new_idx = len(self.rrt_value)
        min_cost = float('inf')
        best_parent = None

        collision_cache = {}

        def cached_is_collision(node1, node2):
            key = tuple(sorted([tuple(node1.round(6)), tuple(node2.round(6))]))
            if key not in collision_cache:
                collision_cache[key] = self.is_collision(node1, node2)
            return collision_cache[key]

        # 第一阶段：为node_new选择父节点
        for idx in neighbor_indices:
            if idx == new_idx:
                continue

            segment_cost = np.linalg.norm(node_new - self.rrt_value[idx])
            potential_cost = self.rrt_cost[idx] + segment_cost

            if potential_cost < min_cost:
                if not cached_is_collision(self.rrt_value[idx], node_new):
                    min_cost = potential_cost
                    best_parent = idx

        # 添加新节点
        rewire_count = 0
        if best_parent is not None:
            self.add_node(node_new, best_parent)
            new_idx = len(self.rrt_value) - 1
        else:
            node_near, idx_near = self.nearest_node(node_new)
            if not cached_is_collision(node_near, node_new):
                self.add_node(node_new, idx_near)
                new_idx = len(self.rrt_value) - 1
                best_parent = idx_near
            else:
                return None, 0

        # 第二阶段：重连接
        for i in neighbor_indices:
            if i == new_idx or i == best_parent:
                continue

            new_segment_cost = np.linalg.norm(node_new - self.rrt_value[i])
            new_total_cost = self.rrt_cost[new_idx] + new_segment_cost

            if new_total_cost < self.rrt_cost[i]:
                if not cached_is_collision(node_new, self.rrt_value[i]):
                    self.rrt_parent[i] = new_idx
                    self.rrt_cost[i] = new_total_cost
                    rewire_count += 1

        return best_parent, rewire_count

    def informed_sample(self):
        """椭圆采样策略"""
        while True:
            x_ball = np.random.uniform(-1, 1, self.dim)
            if np.linalg.norm(x_ball) <= 1:
                break

        # 使用当前最佳路径代价计算椭圆参数
        c = self.c_min / 2
        a = self.best_cost / 2
        self.b = math.sqrt(a ** 2 - c ** 2) if a > c else 1e-6

        # 构建缩放矩阵
        scale_factors = [self.b] * (self.dim - 1) + [a]
        scale_mat = np.diag(scale_factors)

        # 转换到椭圆空间
        x_ellipse = self.x_center + self.C @ scale_mat @ x_ball
        return np.clip(x_ellipse, self.robot.limit_lower, self.robot.limit_upper)

    def random_sample(self):
        """常规随机采样"""
        if np.random.random() > self.sample_rate:
            return self.sample_within_bounds()
        return self.goal

    def sample_within_bounds(self):
        return np.random.uniform(self.robot.limit_lower, self.robot.limit_upper)

    def update_ellipse(self, cost):
        """更新椭圆参数"""
        self.a = cost / 2
        c = self.c_min / 2
        self.b = math.sqrt(self.a ** 2 - c ** 2) if self.a > c else 1e-6

        # 更新旋转矩阵（针对3D空间）
        if self.dim >= 3:
            direction = self.goal[:3] - self.start[:3]
            if np.linalg.norm(direction) > 1e-6:
                rot_vec = np.cross([1, 0, 0], direction / np.linalg.norm(direction))
                theta = np.arccos(np.dot([1, 0, 0], direction / np.linalg.norm(direction)))
                self.C[:3, :3] = R.from_rotvec(rot_vec * theta).as_matrix()

    # 其他方法保持不变...
    def reach_goal(self, node):
        """检查是否到达目标"""
        dist = np.linalg.norm(self.goal - node)
        return dist <= self.step_delta and not self.is_collision(node, self.goal)

    def extract_solution(self):
        # 检查所有节点，找到可到达目标的节点
        goal_indices = []
        for i, node in enumerate(self.rrt_value):
            if self.reach_goal(node):
                goal_indices.append(i)
        if not goal_indices:
            # 尝试添加目标节点
            goal_idx = len(self.rrt_value) - 1
            if np.linalg.norm(self.rrt_value[goal_idx] - self.goal) > 1e-6:
                self.add_node(self.goal, goal_idx)
                goal_idx = len(self.rrt_value) - 1
            config, path, euler = self.extract_path(goal_idx)
            cost = self.rrt_cost[goal_idx]
            return (config, path, euler), cost
        # 从所有目标节点中选择代价最小的路径
        best_cost = float('inf')
        best_path_info = None
        for goal_idx in goal_indices:
            config, path, euler = self.extract_path(goal_idx)
            if np.linalg.norm(config[-1] - self.goal) > 1e-6:
                # 添加最后一段到目标点的路径
                config.append(self.goal.copy())
                pos, orn = self.robot.kinematics(self.goal)
                # 需要添加对应的位置和姿态
                path.append(pos)
                euler.append(orn)
            cost = self.rrt_cost[goal_idx]
            if cost < best_cost:
                best_cost = cost
                best_path_info = (config, path, euler)
        return best_path_info, best_cost

    def new_node(self, start, goal):
        dist = np.linalg.norm(goal - start)
        if dist <= self.step_delta:
            return goal
        else:
            direction = (goal - start) / dist
            return start + direction * self.step_delta

    def nearest_node(self, node):
        if len(self.rrt_value) == 0:
            return None, -1
        dist = np.linalg.norm(np.array(self.rrt_value) - node, axis=1)
        idx = np.argmin(dist)
        return self.rrt_value[idx], idx

    def find_neighbors(self, node):
        if len(self.rrt_value) == 0:
            return []
        dist = np.linalg.norm(np.array(self.rrt_value) - node, axis=1)
        # n = len(self.rrt_value) + 1
        # gamma = 3.0  # 经验系数，可调 2~5
        # radius = gamma * (math.log(n) / n) ** (1.0 / 6.0)
        radius = min(self.search_radius, 50 * self.step_delta)
        return list(np.where(dist < radius)[0])

    def choose_parent(self, node_new, neighbor_indices):
        """选择最优父节点"""
        # 计算所有有效邻居的代价
        valid_indices = []
        valid_costs = []

        for idx in neighbor_indices:
            if not self.is_collision(self.rrt_value[idx], node_new):
                cost = self.rrt_cost[idx] + np.linalg.norm(node_new - self.rrt_value[idx])
                valid_indices.append(idx)
                valid_costs.append(cost)
        # 返回最小代价的父节点索引
        if valid_costs:
            return valid_indices[np.argmin(valid_costs)]
        return None

    def rewire(self, node, neighbors):
        new_idx = len(self.rrt_value) - 1
        for i in neighbors:
            new_cost = self.rrt_cost[new_idx] + np.linalg.norm(node - self.rrt_value[i])
            if new_cost < self.rrt_cost[i] and not self.is_collision(node, self.rrt_value[i]):
                self.rrt_parent[i] = new_idx
                self.rrt_cost[i] = new_cost

    def add_node(self, node, parent_idx):
        self.rrt_value.append(node)
        self.rrt_parent.append(parent_idx)
        pos, orn = self.robot.kinematics(node)
        self.rrt_3Dposition.append(pos)
        self.rrt_3Dorn.append(orn)

        # 计算累积代价
        segment_cost = np.linalg.norm(node - self.rrt_value[parent_idx])
        self.rrt_cost.append(self.rrt_cost[parent_idx] + segment_cost)

    def is_collision(self, start, goal):
        nodes = np.linspace(start, goal, num=max(2, int(np.linalg.norm(goal - start) / self.step_delta * 10)))
        return any(check_all_collision(node, self.robot) for node in nodes)

    def extract_path(self, idx):
        """提取路径"""
        config, path, euler = [], [], []
        current_idx = idx

        while current_idx != -1:
            config.append(self.rrt_value[current_idx])
            path.append(self.rrt_3Dposition[current_idx])
            euler.append(self.rrt_3Dorn[current_idx])
            current_idx = self.rrt_parent[current_idx]

        # 反转路径使其从起点到终点
        return config[::-1], path[::-1], euler[::-1]

    def calculate_path_length(self, path):
        """计算路径长度"""
        if len(path) < 2:
            return 0
        return sum(np.linalg.norm(np.array(path[i]) - np.array(path[i - 1])) for i in range(1, len(path)))

    def visualize(self):
        """可视化结果"""
        # 可视化树结构
        Visualize.plot_points(self.rrt_3Dposition, [1, 0, 0], 4)

        # 可视化边
        edges = []
        for i in range(1, len(self.rrt_parent)):
            if self.rrt_parent[i] != -1:
                edges.append((self.rrt_3Dposition[self.rrt_parent[i]], self.rrt_3Dposition[i]))

        if edges:
            starts = [e[0] for e in edges]
            ends = [e[1] for e in edges]
            Visualize.plot_lines(starts, ends, [0, 1, 0], 1)

        # # 可视化最佳路径
        # if self.best_path:
        #     path = self.best_path[1]
        #     Visualize.plot_lines(path[:-1], path[1:], [1, 0, 1], 3)
        #
        # # 可视化启发式路径
        # if self.heuristic_path and len(self.heuristic_path) > 1:
        #     heuristic_cartesian = []
        #     for joint_point in self.heuristic_path:
        #         pos, _ = self.robot.kinematics(joint_point)
        #         heuristic_cartesian.append(pos)
        #     Visualize.plot_lines(heuristic_cartesian[:-1], heuristic_cartesian[1:], [0, 0.5, 1], 2)
        #     Visualize.plot_points(heuristic_cartesian, [0, 0.5, 1], 8)

# 测试代码
if __name__ == "__main__":
    import pybullet as p
    from robot import DOOSANRobot
    from sim_env.env_dipan import Env
    # 初始化PyBullet环境
    p.connect(p.GUI)
    env =Env()
    # 创建机器人
    robot = DOOSANRobot()

    # 定义起点和终点
    # start_config = [-0.94, 0.628, 1.57, 0, 1.57, 0]
    # # goal_config = [0.94, 0.628, 1.57, 0, 1.57, 0]
    # goal_config = [0, 0.25 * math.pi, 0.45 * math.pi, 0, 0.3 * math.pi, -1.57]

    # start_config = [-0.94, 0.628, 1.57, 0, 1.57, -1.57]
    # goal_config = [0, 0.25 * math.pi, 0.45 * math.pi, 0, 0.3 * math.pi, -1.57]
    start_config = [0, 0.45 * math.pi, 0.25 * math.pi, 0, 0.3 * math.pi, -1.57]
    goal_config = [0, 0, 1.57, 0, 1.57, -1.57]
    # 创建改进的Informed RRT*规划器
    planner = LAPF_InformedRRTStar(
        start=start_config,
        goal=goal_config,
        step_delta=0.3,
        iter_max=5000,
        sample_rate=0.3,
        search_radius=0.5,
        robot=robot,
        use_heuristic=True  # 启用启发式路径
    )
    # 固定机械臂初始位置
    def fix_robot_position(robot, start_config):
        for i, joint_id in enumerate(robot.doosan_joints):
            p.setJointMotorControl2(
                bodyIndex=robot.doosan_id,
                jointIndex=joint_id,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=0,
                force=100  # 足够的力保持位置
            )
            p.resetJointState(robot.doosan_id, joint_id, start_config[i])

        # 稳定机械臂
        for _ in range(100):
            p.stepSimulation()
            time.sleep(0.01)

    # 执行规划
    path = planner.planning(False)
    Visualize.plot_path(path[1], [0.1, 0.1, 0.1], [1, 0, 1], 4, 2)
    # 插值平滑关节角
    raw_configuration = Bspline_2.bspline_smooth_path(path[0])
    # 正运动学求末端位姿
    raw_path = []
    for i in range(len(raw_configuration)):
        path, _ = robot.kinematics(raw_configuration[i])
        raw_path.append(path)
    print(raw_path)
    Visualize.plot_path(raw_path, [0.1, 0.1, 0.1], [0, 0, 1], 4, 2)
    fix_robot_position(robot, planner.start)
    robot.run_path(raw_configuration)  # 使用平滑后的运动
    # 转换为numpy数组以便处理
    config_array = np.array(raw_configuration)
    # 保存到txt文件
    np.savetxt("D:/SoftwareInstallation/python-project/doosan_planning/save_datas/raw_config2.txt", config_array, fmt='%.6f', delimiter=',')
    # 保持仿真运行
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1. / 240.)