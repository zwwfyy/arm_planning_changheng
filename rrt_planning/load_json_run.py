import pybullet as p
import numpy as np
import time
import math
import json
import sys
import os
# 将项目根目录加入 sys.path，确保模块导入正确
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# 导入环境，机器人
from arm_env.robot import DoosanRobot
from arm_env.scene import SceneManager
from visualize import Visualize
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
# 固定机械臂初始位置
def fix_robot_position(robot, start_config):
    for i, joint_id in enumerate(robot.doosan_joint_id):
        p.setJointMotorControl2(
            bodyIndex=robot.doosan_id,
            jointIndex=joint_id,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=100
        )
        p.resetJointState(robot.doosan_id, joint_id, start_config[i])


json_path = "D:\\SoftwareInstallation\\python-project\\RL\\rl_learning\\arm_planning_changheng\\rrt_planning\\path_joint.json"
with open(json_path, "r") as f:
    configuration = json.load(f)
configuration = np.array(configuration)
configuration = np.deg2rad(configuration)
raw_path = []
for i in range(len(configuration)):
    path,_ = robot.kinematics(configuration[i])
    raw_path.append(path)
Visualize.plot_path(raw_path, [0.1, 0.1, 0.1], [0, 0, 1], 4, 2)
fix_robot_position(robot,[-1.57, 0, 1.57, 0, 1.57, -1.57])
time.sleep(1)
robot.run_path(configuration)#使用平滑后的运动
fix_robot_position(robot,configuration[99])#固定机械臂在末尾位置
t = 0
while p.isConnected():
    for _ in range(240 * 10):
        p.stepSimulation()
        time.sleep(1/240)
