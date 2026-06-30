import pybullet as p
import pybullet_data
import numpy as np
import time
import math
class DoosanRobot():

    def __init__(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setAdditionalSearchPath("D:/SoftwareInstallation/python-project/RL/rl_learning/arm_planning/")
        basePosition = [-1.0, -0.5, 0.8]
        baseOrientation = [0, 0, 0, 1]
        self.dipan = p.loadURDF("D:/SoftwareInstallation/python-project/RL/rl_learning/arm_planning/doosan/dipan.urdf", [-1.0, -0.5, 0], [0, 0, 0, 1],useFixedBase=True)
        self.doosan_id = p.loadURDF("D:/SoftwareInstallation/python-project/RL/rl_learning/arm_planning/doosan/h2017.white1.5.1.urdf",
                                    basePosition, baseOrientation, useFixedBase=True)
        
        self.DOF = 6
        self.robotInfomation(self.doosan_id)
        self.max_velocity = np.minimum(self.limit_velocity, np.array([0.8, 0.8, 0.8, 1.0, 1.0, 1.2], dtype=np.float32))

    # 机械臂关节信息
    def robotInfomation(self, robot_id):
        self.doosan_joint_id = []
        self.limit_lower, self.limit_upper, self.limit_effort, self.limit_velocity = [],[],[],[]
        for i in range(p.getNumJoints(robot_id)):
            info = p.getJointInfo(robot_id,i)
            joint_id = info[0]
            joint_name = info[1].decode("utf-8")
            joint_type = info[2]
            print(f"关节ID: {joint_id}, 名称: {joint_name}, 类型: {joint_type}, 关节下限: {info[8]}, "
                  f"关节上限: {info[9]}, 力矩限制: {info[10]}, 速度限制: {info[11]}")
            if joint_type ==p.JOINT_REVOLUTE:
                self.doosan_joint_id.append(joint_id)
                self.limit_lower.append(info[8])
                self.limit_upper.append(info[9])
                self.limit_effort.append(info[10])
                self.limit_velocity.append(info[11])
            if joint_name =="joint_6":
                self.doosan_hand_id = joint_id
        return self.doosan_joint_id, self.doosan_hand_id

    # 重新固定机械臂位置
    def reset_joints(self, joints):
       for i in range(self.DOF):
           p.resetJointState(self.doosan_id, i, joints[i], targetVelocity=0)

    # 下发位置控制指令
    def apply_position_control(self, target_pos):
        p.setJointMotorControlArray(
            bodyUniqueId=self.doosan_id,
            jointIndices=self.doosan_joint_id,
            controlMode=p.POSITION_CONTROL,
            targetPositions=target_pos.tolist(),
            forces=self.limit_effort
        )

    # 下发速度控制指令
    def apply_velocity_control(self, target_vel):
        p.setJointMotorControlArray(
            bodyUniqueId=self.doosan_id,
            jointIndices=self.doosan_joint_id,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=target_vel.tolist(),
            forces=self.limit_effort
        )

    # 获得当前关节状态
    def get_joint_states(self):
        joint_states = p.getJointStates(self.doosan_id, self.doosan_joint_id)
        q = np.array([state[0] for state in joint_states], dtype=np.float32)
        q_v = np.array([state[1] for state in joint_states], dtype=np.float32)
        return q, q_v

    # 正运动学,根据当前关节角求末端位姿
    def get_hand_pos(self):
        state = p.getLinkState(self.doosan_id,self.doosan_hand_id,computeForwardKinematics=True)
        hand_pos = state[0] # state[0] 是 CoM (质心) 位置，state[4] 才是 Link 坐标系原点的位置
        hand_orn = state[1]
        return hand_pos, hand_orn

    #正运动学，输入关节角度求末端工具位姿
    def kinematics(self, configuration):
        for i in range(self.DOF):
            p.resetJointState(self.doosan_id, i, configuration[i])
        link_state = p.getLinkState(self.doosan_id, self.doosan_hand_id, computeForwardKinematics=True)
        pos = link_state[0]
        orn = link_state[1]
        orn = p.getEulerFromQuaternion(orn)
        return pos, orn
    
    # 计算逆运动学
    def compute_ik(self, pos):
        joint_angles = p.calculateInverseKinematics(
            self.doosan_id,
            self.doosan_hand_id,
            pos,
            maxNumIterations=100,
            residualThreshold=0.001
        )
        return joint_angles

    # 随机采样目标点
    def sample_random_joints(self, margin: float = 0.15):
        low = self.limit_lower + margin * (self.limit_upper - self.limit_lower)
        high = self.limit_upper - margin * (self.limit_upper - self.limit_lower)
        return np.random.uniform(low, high).astype(np.float32)
    
    # 计算关节角度距离
    def get_joint_limit_margin(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float32)
        lower_margin = q - self.limit_lower
        upper_margin = self.limit_upper - q
        return np.minimum(lower_margin, upper_margin).astype(np.float32)

    # 按照给定的路径运行机械臂
    def run_path(self, configuration):
        for i in range(len(configuration)):
            target_positions = configuration[i]
            p.setJointMotorControlArray(
                positionGains=[1] * self.DOF,
                bodyUniqueId=self.doosan_id,
                jointIndices=list(range(self.DOF)),
                controlMode=p.POSITION_CONTROL,
                targetPositions=target_positions,
            )
            p.stepSimulation()
            time.sleep(1/200)


if __name__ == '__main__':
    p.connect(p.GUI)
    robot = DoosanRobot()
    robot.reset_joints([0, 0.3 * math.pi, 0.4 * math.pi, 0, 0.3 * math.pi, -0.5 * math.pi])
    pos = robot.get_hand_pos()
    print(pos)

    while p.isConnected():
        p.stepSimulation()
