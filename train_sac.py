import os
import gymnasium as gym
import pybullet as p
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from arm_env.doosan_env import LocalAvoidanceEnv

log_dir = "./logs/sac_log/"
save_dir = "./models/sac_model/"
TOTAL_TIMESTEPS = 2000000


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
    set_random_seed(i)
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
        name_prefix="sac_checkpoint"
    )

    model = SAC(
        policy="MlpPolicy",
        env=env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        buffer_size=1_000_000,          # 经验回放池大小（SAC 是 off-policy，需要大 buffer）
        batch_size=256,                  # 每次更新采样的 batch 大小
        tau=0.005,                       # 软更新系数
        gamma=0.99,
        ent_coef="auto",                 # 自动调节熵系数，鼓励探索
        target_entropy="auto",           # 自动计算目标熵
        train_freq=64,                   # 每收集 64 步更新一次（与 n_steps 类似）
        gradient_steps=64,               # 每次更新做 64 个 gradient step
        learning_starts=1000,            # 随机探索 1000 步后才开始训练
        use_sde=True,                    # 使用广义状态相关探索噪声（连续动作推荐）
        device="cuda"
    )

    print(f"开始训练，总步数: {TOTAL_TIMESTEPS} ...")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        tb_log_name="SAC_Run1",
    )

    model.save(os.path.join(save_dir, "sac_final_model"))
    print("训练完成！最终模型已保存。")

    env.close()


if __name__ == "__main__":
    main()
