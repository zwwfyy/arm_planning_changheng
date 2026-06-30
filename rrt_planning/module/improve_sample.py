import numpy as np
import math
import random
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rot


class JointAdaptiveSampler:
    """
    针对机械臂关节空间的自适应采样器。
    """

    def __init__(self, robot,
                 x_start, x_goal, initial_path=None,
                 narrow_threshold=0.2,   # clearance proxy 阈值（越小越严格）
                 wide_threshold=0.6,
                 min_density=0.1, max_density=1.0,
                 adaptive_ratio=0.8,
                 perturb_sigma=0.05,    # 扰动标准差
                 perturb_samples=30):   # 每个路径点用于估算的扰动采样数
        self.robot = robot
        self.dim = len(x_start)
        self.x_start = np.array(x_start)
        self.x_goal = np.array(x_goal)

        self.narrow_threshold = narrow_threshold
        self.wide_threshold = wide_threshold
        self.min_density = min_density
        self.max_density = max_density
        self.adaptive_ratio = adaptive_ratio

        self.perturb_sigma = perturb_sigma
        self.perturb_samples = int(perturb_samples)

        # 路径/统计信息
        self.initial_path = None if initial_path is None else [np.array(p) for p in initial_path]
        self.path_segment_info = []    # 每个点字典列表
        self.sampling_density_map = {} # map: tuple(config) -> density

        # robot limits
        self.lower = np.array(self.robot.limit_lower)
        self.upper = np.array(self.robot.limit_upper)

        # 若传入初始路径，则立即分析
        if self.initial_path:
            self.set_initial_path(self.initial_path)

    def set_initial_path(self, path):
        """设置初始路径（关节空间路径），并做特征分析"""
        if not path or len(path) < 2:
            print("警告：初始路径无效（长度<2）")
            self.initial_path = None
            self.path_segment_info = []
            self.sampling_density_map = {}
            return
        self.initial_path = [np.array(p) for p in path]
        self.analyze_path_characteristics(self.initial_path)
        print(f"路径特征分析完成，共 {len(self.path_segment_info)} 个路径点")

    def analyze_path_characteristics(self, path):
        """对路径每个内部点构建密度映射"""
        self.path_segment_info = []
        L = len(path)
        for i in range(1, L - 1):
            prev_p = np.array(path[i - 1])
            curr_p = np.array(path[i])
            next_p = np.array(path[i + 1])

            # 关节空间曲率 proxy：使用两个差向量的夹角度量
            v1 = curr_p - prev_p
            v2 = next_p - curr_p
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-8 or n2 < 1e-8:
                curvature = 0.0
            else:
                v1n = v1 / n1
                v2n = v2 / n2
                dot = np.clip(np.dot(v1n, v2n), -1.0, 1.0)
                curvature = 1.0 - dot

            # clearance proxy: 在当前关节点附近做若干小扰动，统计无碰撞比例
            collision_free_count = 0
            for _ in range(self.perturb_samples):
                perturb = np.random.normal(scale=self.perturb_sigma, size=self.dim)
                sample_cfg = curr_p + perturb
                # 裁剪到关节限制
                sample_cfg = np.clip(sample_cfg, self.lower, self.upper)
                if not self._is_collision_config(sample_cfg):
                    collision_free_count += 1
            clearance_score = collision_free_count / max(1, self.perturb_samples)  # 0..1，越大越安全
            obstacle_density = 1.0 - clearance_score  # 越大越拥挤/危险
            is_narrow = clearance_score < self.narrow_threshold
            is_wide = clearance_score > self.wide_threshold

            seg = {
                'point': curr_p,
                'index': i,
                'curvature': float(curvature),
                'clearance_score': float(clearance_score),
                'obstacle_density': float(obstacle_density),
                'is_narrow': bool(is_narrow),
                'is_wide': bool(is_wide)
            }
            self.path_segment_info.append(seg)

        # 根据分析结果构建密度映射
        self.build_sampling_density_map()

    def _is_collision_config(self, cfg):
        """检查给定关节配置是否碰撞"""
        try:
            from utils import check_collision
        except Exception:
            return False
        return check_collision(cfg, self.robot)

    def build_sampling_density_map(self):
        """将上一步的特征映射到采样密度（密度越高表示该区域采样越密集）"""
        self.sampling_density_map = {}
        if not self.path_segment_info:
            return

        for seg in self.path_segment_info:
            curr = seg['point']
            clearance = seg['clearance_score']   # 0..1
            curvature = seg['curvature']
            obstacle_density = seg['obstacle_density']

            # 基础密度：狭窄 -> 高密度，宽阔 -> 低密度
            if seg['is_narrow']:
                base = self.max_density
            elif seg['is_wide']:
                base = self.min_density
            else:
                base = (self.min_density + self.max_density) / 2.0

            # 曲率调整：转角处增强
            curvature_factor = 1.0 + curvature * 2.0
            base *= curvature_factor

            # 障碍物调整（clearance小 -> 增加密度）
            obstacle_factor = 1.0 + obstacle_density * 1.0
            base *= obstacle_factor

            final = float(np.clip(base, self.min_density, self.max_density))
            self.sampling_density_map[tuple(curr.tolist())] = final

    def adaptive_sample(self, c_best, x_center, C):
        use_adaptive = (self.initial_path is not None and
                        self.path_segment_info and
                        np.random.random() < self.adaptive_ratio)

        if use_adaptive:
            return self.path_based_sample()
        else:
            return self.ellipse_sample(c_best, x_center, C)

    def path_based_sample(self):
        """基于路径特征在关节空间做采样（返回 numpy array）"""
        if not self.path_segment_info or not self.sampling_density_map:
            return self.random_sample()

        points = [seg['point'] for seg in self.path_segment_info]
        densities = np.array([self.sampling_density_map.get(tuple(p.tolist()), self.min_density) for p in points])
        if densities.sum() <= 0:
            probs = np.ones(len(points)) / len(points)
        else:
            probs = densities / densities.sum()

        idx = np.random.choice(len(points), p=probs)
        base = points[idx]
        density = densities[idx]

        # 密度越高，扰动越小
        max_pert = max(1e-3, (1.0 / max(density, 1e-6))) * self.perturb_sigma
        perturb = np.random.normal(scale=max_pert, size=self.dim)
        sample_cfg = base + perturb
        sample_cfg = np.clip(sample_cfg, self.lower, self.upper)
        return sample_cfg

    def ellipse_sample(self, c_best, x_center, C):
        """
        在关节空间实现超椭球采样。
        """
        if np.isfinite(c_best):
            # 维度 self.dim，构造半轴：最后一个轴用 a，其他轴用 b
            c = np.linalg.norm(self.x_goal - self.x_start) / 2.0
            a = c_best / 2.0
            b = math.sqrt(a ** 2 - c ** 2) if a > c else 1e-6

            scale = np.array([b] * (self.dim - 1) + [a])
            # 采样单位球内点（dim维）
            while True:
                x_ball = np.random.uniform(-1.0, 1.0, size=(self.dim,))
                if np.linalg.norm(x_ball) <= 1.0:
                    break
            # 转换： x_center + C @ diag(scale) @ x_ball
            scaled = scale * x_ball
            x_rand = x_center + C.dot(scaled)
            x_rand = np.clip(x_rand, self.lower, self.upper)
            return x_rand
        else:
            return self.random_sample()

    def random_sample(self):
        """在关节范围内均匀随机采样"""
        if np.random.random() > 0.3:
            return np.random.uniform(self.lower, self.upper)
        else:
            return np.array(self.x_goal)

    def get_sampling_info(self):
        """返回统计信息"""
        if not self.path_segment_info:
            return None
        narrow = sum(1 for seg in self.path_segment_info if seg['is_narrow'])
        wide = sum(1 for seg in self.path_segment_info if seg['is_wide'])
        avg_density = float(np.mean(list(self.sampling_density_map.values()))) if self.sampling_density_map else 0.0
        return {
            'total_segments': len(self.path_segment_info),
            'narrow_segments': narrow,
            'wide_segments': wide,
            'average_density': avg_density,
            'density_map': self.sampling_density_map
        }

    def visualize_sampling_density(self, ax=None):
        """绘制路径点 index vs density（关节空间的简化可视化）"""
        info = self.get_sampling_info()
        if not info:
            print("无可视化数据")
            return

        indices = [seg['index'] for seg in self.path_segment_info]
        densities = [self.sampling_density_map.get(tuple(seg['point'].tolist()), 0.0) for seg in self.path_segment_info]
        clearance = [seg['clearance_score'] for seg in self.path_segment_info]
        curvature = [seg['curvature'] for seg in self.path_segment_info]

        if ax is None:
            fig, ax = plt.subplots(2, 1, figsize=(10, 6))
            ax0, ax1 = ax
        else:
            ax0 = ax
            ax1 = None

        ax0.plot(indices, densities, '-o', label='sampling density')
        ax0.set_xlabel('path index')
        ax0.set_ylabel('density')
        ax0.legend()
        if ax1 is not None:
            ax1.plot(indices, clearance, '-o', label='clearance_score')
            ax1.plot(indices, curvature, '-x', label='curvature')
            ax1.set_xlabel('path index')
            ax1.set_ylabel('value')
            ax1.legend()
        plt.tight_layout()
        return ax0

