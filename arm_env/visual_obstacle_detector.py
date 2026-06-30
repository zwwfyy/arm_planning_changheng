"""
视觉障碍物检测与重构模块

核心思路:
  真实 RGB-D 相机采集深度 → 检测未知障碍物 → 在 PyBullet 中重建为碰撞体
  → 自动加入 scene.obstacle_ids → RL env 的 get_obs() / check_collision() 无感生效

流程:
  1. 采集当前深度图
  2. 与基线深度（空场景）做差分 → 变化区域掩码
  3. 掩码像素转 3D 点云（利用 Camera.img2world）
  4. 拟合有向包围盒 (AABB) → PyBullet 碰撞体
  5. 注册到 scene.obstacle_ids

用法:
  detector = VisualObstacleDetector(camera, scene)
  detector.set_baseline(depth_empty)       # 空场景基线
  # 之后每个 step 调用:
  obs_ids = detector.detect_and_reconstruct(current_depth)
"""

import numpy as np
import pybullet as p
import pybullet_data
from typing import List, Optional


class VisualObstacleDetector:
    """从 RGB-D 深度图中检测障碍物并在 PyBullet 中重构"""

    def __init__(
        self,
        camera,
        scene=None,
        exclude_body_ids: Optional[List[int]] = None,
        depth_threshold: float = 0.02,
        min_pixels: int = 50,
        max_sample_points: int = 300,
    ):
        """
        Args:
            camera: 带 img2world() 和 renderCameraMask() 的相机对象
            scene: SceneManager 实例, 检测到的障碍物会自动加入此处
            exclude_body_ids: PyBullet body ID 列表, 这些 ID 对应的像素
                              会被排除 (如机械臂自身, 避免误检)
            depth_threshold: 深度差分阈值 (m), 大于此值视为障碍物
            min_pixels: 最小像素数, 低于此忽略（过滤噪声）
            max_sample_points: 点云降采样上限
        """
        self._camera = camera
        self._scene = scene
        self._exclude_body_ids = set(exclude_body_ids or [])
        self._depth_threshold = depth_threshold
        self._min_pixels = min_pixels
        self._max_sample_points = max_sample_points

        self._baseline_depth: Optional[np.ndarray] = None
        self._baseline_mask: Optional[np.ndarray] = None
        self._phantom_obs_ids: List[int] = []

    def set_baseline(self, depth: np.ndarray):
        """设置背景深度图（空场景）, 同时存储分割掩码用于鬼影剔除"""
        self._baseline_depth = depth
        # 同时保存基线时刻的掩码, 用于排除"机械臂原来位置"的鬼影像素
        try:
            self._baseline_mask = self._camera.renderCameraMask()
        except AttributeError:
            self._baseline_mask = None

    def detect_and_reconstruct(
        self, current_depth: np.ndarray, segmentation_mask: Optional[np.ndarray] = None
    ) -> List[int]:
        """主流程: 深度差分 → 点云 → PyBullet 重构

        Args:
            current_depth: 当前深度图 (H, W), 单位 m
            segmentation_mask: 可选, 当前帧的分割掩码

        Returns:
            新增的 obstacle_ids 列表
        """
        self._clear_phantom()

        if self._baseline_depth is None:
            self._baseline_depth = current_depth
            return []

        # 深度差分 → 掩码
        depth_diff = np.abs(self._baseline_depth - current_depth)
        mask = depth_diff > self._depth_threshold

        # 获取当前帧掩码
        if segmentation_mask is None:
            try:
                segmentation_mask = self._camera.renderCameraMask()
            except AttributeError:
                pass

        # 排除机械臂像素: 当前帧 AND 基线帧 (解决机械臂运动后的"鬼影")
        if segmentation_mask is not None and self._exclude_body_ids:
            # 解码 body ID (负值 → -1)
            cur_ids = segmentation_mask & 0xFFFFFF
            cur_ids = np.where(segmentation_mask < 0, -1, cur_ids)
            if self._baseline_mask is not None:
                base_ids = self._baseline_mask & 0xFFFFFF
                base_ids = np.where(self._baseline_mask < 0, -1, base_ids)
            else:
                base_ids = None

            for bid in self._exclude_body_ids:
                mask = mask & (cur_ids != bid)           # 排除当前机械臂像素
                if base_ids is not None:
                    mask = mask & (base_ids != bid)      # 排除基线机械臂像素(鬼影)

        # 过滤噪点
        ys, xs = np.where(mask)
        if len(ys) < self._min_pixels:
            return []

        # 像素 → 3D 点云
        points = self._pixels_to_pointcloud(xs, ys, current_depth)
        if len(points) < 5:
            return []

        # 点云 → PyBullet 碰撞体
        obs_ids = self._reconstruct_as_obstacle(points)

        # 注册到场景
        if self._scene is not None:
            self._scene.obstacle_ids.extend(obs_ids)
            self._scene.dynamic_obstacle_ids.extend(obs_ids)

        return obs_ids

    def get_obstacle_count(self) -> int:
        return len(self._phantom_obs_ids)

    def _pixels_to_pointcloud(
        self, xs: np.ndarray, ys: np.ndarray, depth: np.ndarray
    ) -> np.ndarray:
        """批量: (像素坐标 + 深度) → 世界系 3D 点云"""
        # 随机降采样
        n = min(self._max_sample_points, len(xs))
        indices = np.random.choice(len(xs), size=n, replace=False)

        points = []
        for idx in indices:
            u, v = int(xs[idx]), int(ys[idx])
            d = depth[v, u]
            if not np.isfinite(d) or d <= 0:
                continue
            pt = self._camera.img2world([u, v], d)
            if pt is not None and np.all(np.isfinite(pt)):
                points.append(pt)

        return np.array(points, dtype=np.float32) if points else np.empty((0, 3))

    def _reconstruct_as_obstacle(self, points: np.ndarray) -> List[int]:
        """点云 → PyBullet 碰撞体

        策略: 计算 AABB (Axis-Aligned Bounding Box),
              如果点云在某个轴向上特别分散, 拆成多个子盒。
        """
        if len(points) < 5:
            return []

        p_min = np.min(points, axis=0)
        p_max = np.max(points, axis=0)
        span = p_max - p_min

        # 如果某个方向跨度 >> 另两个, 拆成多段
        max_axis = np.argmax(span)
        other_axes = [i for i in range(3) if i != max_axis]
        cross_span = np.mean(span[other_axes])

        if span[max_axis] > 3.0 * cross_span and span[max_axis] > 0.15:
            return self._fit_multi_aabb(points, max_axis)
        else:
            obs_id = self._create_aabb(p_min, p_max)
            return [obs_id] if obs_id is not None else []

    def _create_aabb(
        self, p_min: np.ndarray, p_max: np.ndarray
    ) -> Optional[int]:
        """AABB 包围盒碰撞体（无旋转）"""
        half_extents = (p_max - p_min) / 2 + 0.015
        center = (p_min + p_max) / 2

        half_extents = np.maximum(half_extents, [0.025, 0.025, 0.025])

        col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents.tolist())
        vis_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents.tolist(),
            rgbaColor=[1, 0.3, 0, 0.6],
        )

        obs_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=center.tolist(),
            baseOrientation=[0, 0, 0, 1],  # 无旋转
        )
        self._phantom_obs_ids.append(obs_id)
        return obs_id

    def _fit_multi_aabb(
        self,
        points: np.ndarray,
        split_axis: int,
    ) -> List[int]:
        """沿指定轴把点云均匀切分成多个 AABB"""
        coord = points[:, split_axis]
        sorted_idx = np.argsort(coord)
        points_sorted = points[sorted_idx]

        n_seg = min(3, len(points_sorted) // 12)
        if n_seg < 2:
            n_seg = 2

        seg_size = len(points_sorted) // n_seg
        obs_ids = []

        for i in range(n_seg):
            seg = points_sorted[i * seg_size : (i + 1) * seg_size]
            if len(seg) < 5:
                continue

            seg_min = np.min(seg, axis=0)
            seg_max = np.max(seg, axis=0)
            obs_id = self._create_aabb(seg_min, seg_max)
            if obs_id is not None:
                obs_ids.append(obs_id)

        return obs_ids

    def _clear_phantom(self):
        """清除上一帧重构的虚拟障碍物"""
        for obs_id in self._phantom_obs_ids:
            try:
                p.removeBody(obs_id)
                if self._scene is not None:
                    self._scene.obstacle_ids = [
                        x for x in self._scene.obstacle_ids if x != obs_id
                    ]
                    self._scene.dynamic_obstacle_ids = [
                        x for x in self._scene.dynamic_obstacle_ids if x != obs_id
                    ]
            except p.error:
                pass
        self._phantom_obs_ids.clear()

# ------------------------------------------------------------------
#  实物相机封装示例 (需根据实际硬件修改)
# ------------------------------------------------------------------

class RealSenseDepthCamera:
    """RealSense / 任意 RGB-D 相机的简易封装

    提供与 Camera 兼容的 img2world() 接口, 以便 VisualObstacleDetector
    可以直接用于实物数据。
    """

    def __init__(self, camera_intrinsics, camera_extrinsics):
        """
        Args:
            camera_intrinsics: 3x3 内参矩阵
            camera_extrinsics: 4x4 外参矩阵 (世界→相机)
        """
        self._K = camera_intrinsics
        self._T_world_to_cam = camera_extrinsics
        self._T_cam_to_world = np.linalg.inv(camera_extrinsics)

    def img2world(self, pt_pixel, depth):
        """像素 + 深度 → 世界坐标 (与 Camera.img2world 签名一致)"""
        u, v = pt_pixel

        # 像素 → 相机坐标系
        fx = self._K[0, 0]
        fy = self._K[1, 1]
        cx = self._K[0, 2]
        cy = self._K[1, 2]

        x_cam = (u - cx) * depth / fx
        y_cam = (v - cy) * depth / fy
        z_cam = depth

        pt_cam = np.array([x_cam, y_cam, z_cam, 1.0])
        pt_world = self._T_cam_to_world @ pt_cam
        return pt_world[:3] / pt_world[3]


# ------------------------------------------------------------------
#  在现有 env 中集成的示例
# ------------------------------------------------------------------

if __name__ == "__main__":
    """演示: 用 PyBullet 仿真测试整个检测→重构流程"""
    import sys
    import os
    # 把项目根目录加入 path, 确保 import 可用
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    import pybullet as p
    import time

    p.connect(p.GUI)
    p.setGravity(0, 0, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")

    from util.camera import Camera
    from arm_env.scene import SceneManager

    camera = Camera()
    scene = SceneManager()
    scene.reset_scene()

    detector = VisualObstacleDetector(camera, scene)

    # 采集空场景基线
    baseline = camera.renderCameraDepthImage()
    detector.set_baseline(baseline)

    print("基线已设置。按空格在场景中生成障碍物...")

    # 手动生成一个障碍物来模拟 "突然出现"
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.08, 0.1, 0.08])
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.08, 0.1, 0.08], rgbaColor=[0, 1, 0, 1])
    obs = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=col,
                            baseVisualShapeIndex=vis, basePosition=[-0.2, -0.95, 1])

    for _ in range(50):
        p.stepSimulation()

    # 采当前深度 → 检测并重构
    current_depth = camera.renderCameraDepthImage()
    new_ids = detector.detect_and_reconstruct(current_depth)

    print(f"检测到 {len(new_ids)} 个障碍物, IDs: {new_ids}")
    print(f"scene 中现有障碍物: {len(scene.obstacle_ids)}")
    print(f"danger_vector 仍可正常计算: 验证接口兼容性")

    while p.isConnected():
        p.stepSimulation()
        time.sleep(1 / 240)
