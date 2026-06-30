import sys
import os
# 将项目根目录加入 sys.path，确保模块导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pybullet as p
import numpy as np
import time
import math
# 导入环境，机器人
from arm_env.robot import DoosanRobot
from arm_env.scene import SceneManager
# 导入路径规划相关的
from rrtManipulator import RRT
from LAPF_informrrtstartManipulator_2_pis import LAPF_InformedRRTStar

from visualize import Visualize
# 导入曲线平滑相关
import Bspline_2
np.random.seed(123)

# 初始化环境类
p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0)
env = SceneManager()
env.reset_scene()
robot = DoosanRobot()
p.resetDebugVisualizerCamera(
    cameraDistance=2,
    cameraYaw=45,
    cameraPitch=-30,
    cameraTargetPosition=[0.5,-0.5,1]
)
# 设置初始和目标点
doosan_start = [-1.57, 0, 1.57, 0, 1.57, -1.57]
doosan_end = [0, 0.3 * math.pi, 0.4 * math.pi, 0, 0.3 * math.pi, -0.5 * math.pi]

# 初始化路径规划算法z
rrt = RRT(
    start=doosan_start,
    goal=doosan_end,
    step_delta=0.1,
    iter_max=10000,
    sample_rate=0.9,
    robot=robot,
)
lapf_informrrtstar = LAPF_InformedRRTStar(
    start=doosan_start,
    goal=doosan_end,
    step_delta=0.3,
    iter_max=1000,
    sample_rate=0.3,
    search_radius=0.5,
    robot=robot,
    use_heuristic=True  # 启用启发式路径
)

# 固定机械臂初始位置
def fix_robot_position(robot, start_config):
    for i, joint_id in enumerate(robot.doosan_joint_id):
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

#获取路径
configuration, path, euler= lapf_informrrtstar.planning(plot=False)
print(len(configuration))

raw_configuration = Bspline_2.bspline_smooth_path(configuration)

# 正运动学求末端位姿
raw_path = []
for i in range(len(raw_configuration)):
    path,_ = robot.kinematics(raw_configuration[i])
    raw_path.append(path)

Visualize.plot_path(raw_path, [0.1, 0.1, 0.1], [0, 0, 1], 4, 2)
# # 固定机械臂到开始位置
fix_robot_position(robot, rrt.start)
# 控制机械臂运动
robot.run_path(raw_configuration)#使用平滑后的运动
while p.isConnected():
    p.stepSimulation()
