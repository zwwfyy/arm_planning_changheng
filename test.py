import time

from arm_env.doosan_env import LocalAvoidanceEnv
from stable_baselines3 import SAC,PPO
import pybullet as p
from arm_env.scene import SceneManager
from visualize import Visualize

env = LocalAvoidanceEnv(gui=True)
env.render()
time.sleep(2)
scen = SceneManager()

model = PPO.load("models/1/best_model.zip")
vis = Visualize()
path = []
success_time = 0
obs, info = env.reset()
done = False
step = 0
while not done:
    action,_ = model.predict(obs, deterministic=True)
    obs, reward, terminted, truncated, info = env.step(action)
    path.append(obs[12:15])
    done = terminted or truncated
    step +=1
    time.sleep(0.02)
    if terminted:
        print("避障成功，到达目的地")
        # success_time +=1
# print(f"成功率：{success_time/200}")
vis.plot_path(path,[0.1, 0.1, 0.1], [0, 0, 1], 4, 2)
while p.isConnected():
    p.stepSimulation()



