import datetime
import os

import hydra
import numpy as np
import pytz
import supersuit as ss
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pettingzoo.butterfly import cooperative_pong_v5
from stable_baselines3 import PPO

import wandb
from agent import Agent
from recoder import VideoRecorder
from wandb.integration.sb3 import WandbCallback


class Workspace:
    def __init__(self, cfg: DictConfig):
        japan_tz = pytz.timezone("Japan")
        now = datetime.datetime.now(japan_tz)

        self.run_name = f"{now.strftime('%m_%d_%H:%M')}"
        self.cfg = cfg
        self.model_path = cfg.model_dir + cfg.model_name
        self.run = wandb.init(
            project=cfg.project_name,
            sync_tensorboard=True,
            monitor_gym=True,
            name=self.run_name,
        )

        root_dir = os.getcwd()
        self.recoder = VideoRecorder(root_dir)

    def batchify_obs(obs, device):
        """Converts PZ style observations to batch of torch arrays."""
        # convert to list of np arrays
        obs = np.stack([obs[a] for a in obs], axis=0)
        # transpose to be (batch, channel, height, width)
        obs = obs.transpose(0, -1, 1, 2)
        # convert to torch
        obs = torch.tensor(obs).to(device)

        return obs

    def batchify(x, device):
        """Converts PZ style returns to batch of torch arrays."""
        # convert to list of np arrays
        x = np.stack([x[a] for a in x], axis=0)
        # convert to torch
        x = torch.tensor(x).to(device)

        return x

    def unbatchify(x, env):
        """Converts np array to PZ style arguments."""
        x = x.cpu().numpy()
        x = {a: x[i] for i, a in enumerate(env.possible_agents)}

        return x

    def train(self):
        # TODO: log training video
        callback = WandbCallback(
            verbose=3,
            model_save_path=self.cfg.model_dir,
            model_save_freq=100,
        )

        # TODO: log cumulative reward and episode length https://pettingzoo.farama.org/api/utils/
        env = cooperative_pong_v5.parallel_env(
            render_mode="rgb_array",
        )
        env = ss.color_reduction_v0(env)
        num_actions = env.action_space(env.possible_agents[0])
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)
        env = ss.pettingzoo_env_to_vec_env_v1(env)
        env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")

        agent = Agent(num_actions)
        optimizer = optim.Adam(agent.parameters(), lr=0.001, eps=1e-5)
        # model = PPO(
        #     CnnPolicy,
        #     env,
        #     verbose=2,
        #     gamma=self.cfg.gamma,
        #     n_steps=self.cfg.n_steps,
        #     ent_coef=self.cfg.ent_coef,
        #     vf_coef=self.cfg.vf_coef,
        #     learning_rate=self.cfg.learning_rate,
        #     max_grad_norm=self.cfg.max_grad_norm,
        #     gae_lambda=self.cfg.gae_lambda,
        #     n_epochs=self.cfg.n_epochs,
        #     clip_range=self.cfg.clip_range,
        #     batch_size=self.cfg.batch_size,
        # )
        # model.learn(total_timesteps=self.cfg.total_timesteps, callback=callback)
        # model.save(self.model_path)

    def eval(self):
        model = PPO.load(self.model_path)

        env = cooperative_pong_v5.env(render_mode="rgb_array")
        env = ss.color_reduction_v0(env)
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 3)

        for eval_episode in range(self.cfg.eval_episodes):
            self.recoder.init()
            env.reset()
            video_path = f"latest_{eval_episode}.mp4"
            for agent in env.agent_iter():
                observation, reward, done, truncation, info = env.last()

                if done or truncation:
                    action = None
                else:
                    action = (
                        model.predict(observation, deterministic=True)[0]
                        if not done
                        else None
                    )

                env.step(action)
                self.recoder.record(env)
            self.recoder.save(video_path)
            self.run.log({"video": wandb.Video(f"video/{video_path}", fps=30)})
        self.run.finish()
        env.close()


@hydra.main(version_base=None, config_path="config", config_name="pong")
def main(cfg: DictConfig):
    workspace = Workspace(cfg)
    if cfg.train:
        workspace.train()
    workspace.eval()


if __name__ == "__main__":
    main()
