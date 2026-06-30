import pybullet as p
import pybullet_data
import numpy as np
import random


class SceneManager:
    def __init__(self, workspace_limits: dict = None):
        if workspace_limits is None:
            self.workspace_limits = {
                'x': [0.1, 0.6],   # 机械臂前方范围
                'y': [-0.4, 0.4],  # 左右范围
                'z': [0.05, 0.6]   # 高度范围
            }
        else:
            self.workspace_limits = workspace_limits
        
        # 记录当前场景中的所有障碍物ID
        self.obstacle_ids = []
        self.static_obstacle_ids = []
        self.dynamic_obstacle_ids = []
        self._setup_basic_scene()

    def _setup_basic_scene(self):
        """初始化重力和地面。"""
        p.setGravity(0, 0, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf")

    def reset_scene(self):
        """重置场景，并重新生成静态障碍物。"""
        self.clear_all_obstacles()
        tableUid = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.5, -0.5, -0.04],
            baseOrientation=[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
        )

        # 创建门框和凸起障碍物
        SILVER_COLOR = [0.85, 0.85, 0.9, 1.0]
        kuang1 = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.5]
        )
        # 门框左侧立柱
        base_position = [0.14, -1.04, 1.08]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        body_id1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=kuang1,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        p.changeVisualShape(body_id1, -1, rgbaColor=SILVER_COLOR)

        # 左侧凸起
        tuqi1 = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.005, 0.025]
        )
        base_position = [0.05, -1.08, 0.862]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        tuqi_body1 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tuqi1,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        p.changeVisualShape(tuqi_body1, -1, rgbaColor=SILVER_COLOR)

        # 门框右侧立柱
        kuang2 = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[0.04, 0.04, 0.5]
        )
        base_position = [0.14, 0.04, 1.08]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        body_id2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=kuang2,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        p.changeVisualShape(body_id2, -1, rgbaColor=SILVER_COLOR)
        kuang3 = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[0.04, 0.58, 0.04]
        )
        # 右侧凸起
        tuqi2 = p.createCollisionShape(
            shapeType=p.GEOM_BOX, halfExtents=[0.05, 0.005, 0.025]
        )
        base_position = [0.05, 0.08, 0.862]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        tuqi_body2 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=tuqi2,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        p.changeVisualShape(tuqi_body2, -1, rgbaColor=SILVER_COLOR)
        # 门框顶部横梁
        base_position = [0.14, -0.5, 1.62]
        base_orientation = p.getQuaternionFromEuler([0, 0, 0])
        body_id3 = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=kuang3,
            basePosition=base_position,
            baseOrientation=base_orientation,
        )
        p.changeVisualShape(body_id3, -1, rgbaColor=SILVER_COLOR)
        self.obstacleID = [tableUid, body_id1, body_id2, body_id3, tuqi_body1, tuqi_body2]
        self.obstacle_ids.extend(self.obstacleID)
        self.static_obstacle_ids.extend(self.obstacleID)

    def clear_all_obstacles(self):
        """清空当前场景中记录的所有障碍物。"""
        for obs_id in self.obstacle_ids:
            try:
                p.removeBody(obs_id)
            except p.error:
                pass
        self.obstacle_ids.clear()
        self.static_obstacle_ids.clear()
        self.dynamic_obstacle_ids.clear()

    def _create_random_shape(self, pos):
        """在指定位置创建一个随机形状的静态障碍物。"""
        shape_type = random.choice(['sphere'])
        if shape_type == 'sphere':
            radius = random.uniform(0.08, 0.11)
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[0, 0, 1, 1.0])
        elif shape_type == 'box':
            hx, hy, hz = [random.uniform(0.03, 0.06) for _ in range(3)]
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0, 0, 1, 1.0])
        else:
            radius = random.uniform(0.03, 0.06)
            length = random.uniform(0.1, 0.2)
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[0, 0, 1, 1.0])

        random_quat = p.getQuaternionFromEuler([random.uniform(0, 2 * np.pi) for _ in range(3)])
        obs_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col_id, baseVisualShapeIndex=vis_id,
            basePosition=pos.tolist(), baseOrientation=random_quat
        )
        return obs_id

    def _respawn_obstacle(self):
        """在指定的 X-Y-Z 矩形区域内随机生成一个动态障碍物。"""
        corner_min = np.array([-0.3, -1.1, 1.00])
        corner_max = np.array([-0.1, -0.8, 1.28])
        obs_pos = np.array([
            random.uniform(corner_min[0], corner_max[0]),
            random.uniform(corner_min[1], corner_max[1]),
            random.uniform(corner_min[2], corner_max[2]),
        ])
        shape_type = random.choice(['sphere', 'box', 'cylinder'])
        
        if shape_type == 'sphere':
            radius = random.uniform(0.1, 0.15)
            col_id = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
            vis_id = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=[1, 0, 0, 1.0])
        
        elif shape_type == 'box':
            hx, hy, hz = [random.uniform(0.1, 0.15) for _ in range(3)]
            col_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz])
            vis_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[1, 0, 0, 1.0])
        
        else:
            radius = random.uniform(0.1, 0.15)
            length = random.uniform(0.1, 0.15)
            col_id = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
            vis_id = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=[1, 0, 0, 1.0])
        
        # 暂时固定朝向，避免随机旋转引入额外不稳定性。
        random_quat = [0, 0, 0, 1]
        obs_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=col_id,
            baseVisualShapeIndex=vis_id,
            basePosition=obs_pos.tolist(),
            baseOrientation=random_quat
        )
        self.obstacle_ids.append(obs_id)
        self.dynamic_obstacle_ids.append(obs_id)


if __name__ == '__main__':
    p.connect(p.GUI)
    scene = SceneManager()
    scene.reset_scene()
    # scene._respawn_obstacle()

    while p.isConnected():
        p.stepSimulation()


