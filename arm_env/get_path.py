import numpy as np
# 导入路径规划相关的
from rrt_planning.rrtManipulator import RRT
np.random.seed(123)

class GetPath:
    def __init__(self, robot, doosan_start, dooosan_end):
        self.doosan_start = doosan_start
        self.doosan_end = dooosan_end
        self.robot = robot
        self.rrt = RRT(start=self.doosan_start, goal=self.doosan_end, step_delta=0.1, iter_max=10000,sample_rate=0.9, robot=self.robot)

    def get_configuration(self):
        result = self.rrt.planning(plot=False)
        return result