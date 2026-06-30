import numpy as np
from visualize import Visualize
import time
# check_collision 根据任务可替换
from util.utils import check_all_collision


class RRT:
    """
    RRT (Rapidly-exploring Random Trees) 算法实现，用于机械臂运动规划
    """

    def __init__(self, start, goal, step_delta, iter_max, sample_rate, robot):
        """
        初始化RRT规划器

        参数:
            start: 起点配置（弧度）
            goal: 目标配置（弧度）
            step_delta: 每次扩展的最大步长
            iter_max: 最大迭代次数
            sample_rate: 采样目标点的概率
            robot: 机器人模型，需包含运动学和关节限制信息
        """
        self.start = np.array(start)
        self.goal = np.array(goal)
        self.step_delta = step_delta  # 步长参数，控制树的扩展速度
        self.iter_max = iter_max  # 最大迭代次数，限制算法运行时间
        self.sample_rate = sample_rate  # 目标偏向采样概率，多大概率以终点作为随机采样点
        self.robot = robot  # 机器人模型

        # RRT树的核心数据结构
        # rrt_value: 存储所有节点的关节配置
        # rrt_parent: 存储每个节点的父节点索引
        # rrt_3Dposition: 存储每个节点的末端执行器3D位置
        self.rrt_value = [self.start]
        self.rrt_parent = [-1]  # -1表示根节点（起点）
        pos, orn = self.robot.kinematics(start)
        self.rrt_3Dposition = [pos]
        self.rrt_3Dorn = [orn]

    def calculate_path_length(self, path_3d):
        """
        计算路径在3D空间中的长度（末端执行器的轨迹长度）

        参数:
            path_3d: 3D位置路径

        返回:
            路径的总长度（米）
        """
        length = 0.0
        for i in range(1, len(path_3d)):
            # 计算相邻两点之间的欧氏距离
            length += np.linalg.norm(np.array(path_3d[i]) - np.array(path_3d[i - 1]))
        return length

    # 主规划函数
    def planning(self, plot=True):
        """
        执行RRT规划

        参数:
            plot: 是否可视化规划过程

        返回:
            如果找到路径，返回配置路径和3D位置路径；否则返回None
        """
        start_time = time.time()  # 记录开始时间
        for i in range(self.iter_max):
            # 1. 生成随机节点
            node_rand = self.generate_random_node()
            # 2. 找到最近节点
            node_near, index_near = self.nearest_node(node_rand)
            # 3. 生成新节点
            node_new = self.new_node(node_near, node_rand)

            # 4. 碰撞检测
            if not self.is_collision(node_near, node_new):
                # 5. 添加新节点到树
                self.rrt_append(node_new, index_near)
                # 6. 检查是否接近目标
                dist = np.linalg.norm(self.goal - node_new, 2)

                # 7. 如果接近目标且无碰撞，连接到目标并返回路径
                if dist <= self.step_delta and not self.is_collision(
                        node_new, self.goal
                ):
                    end_time = time.time()  # 记录结束时间
                    planning_time = end_time - start_time
                    if plot:
                        self.plot_points_and_lines()
                    # 添加目标节点到树
                    self.rrt_append(self.goal, len(self.rrt_value) - 2)
                    # 提取从起点到目标的完整路径
                    configuration, path, euler = self.extract_path(len(self.rrt_value) - 1)
                    # 计算路径长度
                    path_length = self.calculate_path_length(path)
                    # 打印规划时间和路径长度
                    return configuration, path, euler
    # 生成随机节点（带目标偏向策略）
    def generate_random_node(self):
        """
        生成随机节点，以一定概率直接采样目标点

        返回:
            随机采样的节点或目标节点
        """
        if np.random.random() > self.sample_rate:
            return self.sample_within_bounds()  # 在关节限制内随机采样
        else:
            return self.goal  # 目标偏向采样

    # 生成新节点（从near向rand扩展固定步长）
    def new_node(self, start, goal):
        """
        从start向goal扩展固定步长生成新节点

        参数:
            start: 起点节点
            goal: 目标节点

        返回:
            扩展后的新节点
        """
        # 计算两点间的欧氏距离
        dist = np.linalg.norm(goal - start, 2)
        assert self.step_delta > 0

        # 如果距离超过步长，按步长比例扩展；否则直接使用目标点
        if dist > self.step_delta:
            return start + (goal - start) / dist * self.step_delta
        else:
            return goal

    # 在关节限制范围内随机采样
    def sample_within_bounds(self):
        """
        在机器人关节限制范围内随机采样节点

        返回:
            随机采样的有效关节配置
        """
        limit_lower = self.robot.limit_lower  # 关节下限
        limit_upper = self.robot.limit_upper  # 关节上限
        assert len(limit_lower) == len(limit_upper)

        # 在每个关节的限制范围内均匀随机采样
        node = np.random.uniform(limit_lower, limit_upper)
        return node

    # 找到树中距离给定点最近的节点
    def nearest_node(self, node):
        """
        找到RRT树中距离给定点最近的节点

        参数:
            node: 目标节点

        返回:
            最近节点及其索引
        """
        # 计算所有节点到目标节点的欧氏距离
        dist = np.linalg.norm(np.array(self.rrt_value) - node, axis=1)
        nearest_index = np.argmin(dist)  # 最小距离对应的索引
        return self.rrt_value[nearest_index], nearest_index

    # 在两点之间生成均匀分布的采样点
    def path_nodes(self, start, goal, step_collision):
        """
        在两点之间生成均匀分布的采样点，用于碰撞检测

        参数:
            start: 起点
            goal: 终点
            step_collision: 采样步长

        返回:
            采样点数组
        """
        dist = np.linalg.norm(goal - start, 2)
        assert step_collision > 0

        # 计算需要的采样点数
        lin_num = int(np.floor(dist / step_collision))

        # 生成均匀分布的采样点
        if lin_num == 0:
            nodes = np.array([start, goal])
        else:
            nodes = np.linspace(start, goal, lin_num)
        return nodes

    # 检测两点之间的路径是否碰撞
    def is_collision(self, start, goal):
        """
        检测从start到goal的路径是否与障碍物碰撞

        参数:
            start: 起点
            goal: 终点

        返回:
            如果碰撞返回True，否则返回False
        """
        # 设置碰撞检测的采样步长
        step_collision = self.step_delta / 10
        # 生成路径上的采样点
        nodes = self.path_nodes(start, goal, step_collision)

        # 对每个采样点进行碰撞检测
        for node in nodes:
            if check_all_collision(node, self.robot):
                return True
        return False

    # 从终点回溯提取完整路径
    def extract_path(self, index_end):
        """
        从终点索引回溯到起点，提取完整路径

        参数:
            index_end: 终点索引

        返回:
            关节配置路径和对应的3D位置路径
        """
        configuration = []  # 关节配置路径
        path = []  # 3D位置路径
        euler = []
        index_now = index_end  # 从终点开始回溯

        # 沿父节点指针回溯到起点
        while index_now != -1:
            configuration.append(self.rrt_value[index_now])
            path.append(self.rrt_3Dposition[index_now])
            euler.append(self.rrt_3Dorn[index_now])
            index_now = self.rrt_parent[index_now]

        # 返回从起点到终点的路径（需要反转）
        return configuration[::-1], path[::-1], euler[::-1]

    # 向RRT树添加新节点
    def rrt_append(self, node, index):
        """
        向RRT树添加新节点

        参数:
            node: 新节点
            index: 父节点索引
        """
        self.rrt_value.append(node)  # 添加节点配置
        self.rrt_parent.append(index)  # 记录父节点
        # 计算并存储新节点的3D位置
        pos, orn = self.robot.kinematics(node)
        self.rrt_3Dposition.append(pos)
        self.rrt_3Dorn.append(orn)

    # 可视化RRT树（点和边）
    def plot_points_and_lines(self):
        """
        可视化RRT树的节点和边
        """
        # 绘制所有节点（红色）
        Visualize.plot_points(self.rrt_3Dposition, [1, 0, 0], 4)

        # 提取所有边的起点和终点
        end = self.rrt_3Dposition[1:]  # 边的终点
        begin_indices = self.rrt_parent[1:]  # 边的起点索引
        begin = [self.rrt_3Dposition[i] for i in begin_indices]  # 边的起点

        # 绘制所有边（绿色）
        Visualize.plot_lines(begin, end, [0, 1, 0], 2)