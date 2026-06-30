import os
import gymnasium as gym
import pybullet as p
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv,SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from arm_env.doosan_env import LocalAvoidanceEnv
log_dir = "./logs/pos_log/"
save_dir = "./models/pos_model/"

def make_env(i):
    def _init():
        if i == 0:
            env = LocalAvoidanceEnv(gui=True)
        else:
            env = LocalAvoidanceEnv(gui=False)
        env = Monitor(env, log_dir)
        env.render()
        env.reset()
        return env
    set_random_seed(i)  # 每个子环境不同种子，避免完全相同的随机序列
    return _init


def main():
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    print("正在初始化仿真环境...")


    num_train = 4
    env = SubprocVecEnv([make_env(i) for i in range(num_train)])


    eval_callback = EvalCallback(
        env,
        best_model_save_path=save_dir,
        log_path=log_dir,
        eval_freq=5000,
        n_eval_episodes=5,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=20000,
        save_path=save_dir,
        name_prefix="ppo_checkpoint"
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        n_steps=2048,              # 每个环境收集2048步 → 总rollout = 4×2048 = 8192步/次更新
        batch_size=256,            # 8192/256 = 32个mini-batch，比128更稳定
        n_epochs=10,               # 每个batch反复优化10轮
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        device="cuda"
    )

    total_timesteps = 2000000
    print(f"开始训练，总步数: {total_timesteps} ...")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="PPO_MVP_Run1",
    )

    model.save(os.path.join(save_dir, "ppo_final_model"))
    print("训练完成！最终模型已保存。")

    env.close()


if __name__ == "__main__":
    main()