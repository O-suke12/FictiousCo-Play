import datetime
import os

import hydra
import pytz
import supersuit as ss
from omegaconf import DictConfig
from pettingzoo.butterfly import cooperative_pong_v5
from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy

import wandb
from recoder import VideoRecorder
from wandb.integration.sb3 import WandbCallback

japan_tz = pytz.timezone("Japan")
now = datetime.datetime.now(japan_tz)


def train(cfg: DictConfig):
    run_name = f"{cfg.project_name}__{now.strftime('%Y-%m-%d--%H-%M')}"
    run = wandb.init(
        project=cfg.project_name,
        sync_tensorboard=True,
        monitor_gym=True,
        name=run_name,
    )

    callback = WandbCallback(
        verbose=3,
        model_save_path="models/",
        model_save_freq=100,
    )

    env = cooperative_pong_v5.parallel_env(
        render_mode="rgb_array",
    )
    env = ss.color_reduction_v0(env)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")

    observation = env.reset()
    model = PPO(
        CnnPolicy,
        env,
        verbose=3,
        gamma=0.95,
        n_steps=256,
        ent_coef=0.0905168,
        learning_rate=0.00062211,
        vf_coef=0.042202,
        max_grad_norm=0.9,
        gae_lambda=0.99,
        n_epochs=5,
        clip_range=0.3,
        batch_size=256,
        tensorboard_log="runs",
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=callback)
    model.save(cfg.model_path + cfg.model)
    run.finish()


def eval(cfg):
    root_dir = os.getcwd()
    recoder = VideoRecorder(root_dir)
    recoder.init()

    model = PPO.load(cfg.model_path + cfg.model)

    env = cooperative_pong_v5.env(render_mode="rgb_array")
    env = ss.color_reduction_v0(env)
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 3)
    env.reset()

    for agent in env.agent_iter():
        observation, reward, done, truncation, info = env.last()

        if done or truncation:
            action = None
        else:
            action = (
                model.predict(observation, deterministic=True)[0] if not done else None
            )

        env.step(action)
        recoder.record(env)
    recoder.save("video.mp4")
    env.close()


@hydra.main(config_path="config", config_name="pong")
def main(cfg: DictConfig):
    train(cfg)
    eval(cfg)


if __name__ == "__main__":
    main()
