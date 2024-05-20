import datetime
import os

import hydra
import numpy as np
import pytz
from omegaconf import DictConfig
from pettingzoo.mpe import simple_spread_v3
from stable_baselines3 import PPO

import wandb
from utils.recoder import VideoRecorder
from utils.replay_memory import ReplayMemory
from rili.rili import RILI


class Workspace:
    def __init__(self, cfg: DictConfig):
        japan_tz = pytz.timezone("Japan")
        now = datetime.datetime.now(japan_tz)

        self.run_name = f"{now.strftime('%m_%d_%H:%M')}"
        self.cfg = cfg
        self.model_path = cfg.model_dir + cfg.model_name
        # self.run = wandb.init(
        #     project=cfg.project_name,
        #     sync_tensorboard=True,
        #     monitor_gym=True,
        #     name=self.run_name,
        # )

        root_dir = os.getcwd()
        self.recoder = VideoRecorder(root_dir)

        self.env = simple_spread_v3.env(
            N=2, local_ratio=0.5, max_cycles=25, continuous_actions=False
        )
        self.env.reset()

        agent_name = self.env.agents[0]
        action_space = int(self.env.action_space(agent_name).n)
        observation_space = self.env.observation_space(agent_name).shape[0]

        self.agent = RILI(action_space, observation_space, self.cfg.max_episode_steps)
        self.memory = ReplayMemory(
            capacity=self.cfg.num_eps, interaction_length=cfg.max_episode_steps
        )

    def train(self):
        z_prev = np.zeros(10)
        z = np.zeros(10)
        for i_episode in range(1, self.cfg.num_eps + 1):
            if len(self.memory) > 4:
                z = self.agent.predict_latent(
                    self.memory.get_steps(self.memory.position - 4),
                    self.memory.get_steps(self.memory.position - 3),
                    self.memory.get_steps(self.memory.position - 2),
                    self.memory.get_steps(self.memory.position - 1),
                )

            episode_reward = 0
            episode_steps = 0
            done = False
            state = self.env.reset()

            while not done:
                if i_episode < self.cfg.start_eps:
                    action = self.env.action_space(self.env.agents[0]).sample()
                else:
                    action = self.agent.select_action(state, z)

                if len(self.memory) > self.cfg.batch_size:
                    (
                        critic_1_loss,
                        critic_2_loss,
                        policy_loss,
                        ae_loss,
                        curr_loss,
                        next_loss,
                        kl_loss,
                    ) = self.agent.update_parameters(self.memory, self.cfg.batch_size)

                next_state, reward, done, _ = self.env.step(action)
                episode_steps += 1
                episode_reward += reward

                mask = (
                    1
                    if episode_steps == self.env._max_episode_steps
                    else float(not done)
                )
                self.memory.push_timestep(state, action, reward, next_state, mask)
                state = next_state

    def eval(self):
        model = PPO.load(self.model_path)

        env = simple_spread_v3.env(
            max_cycles=500,
            x_size=16,
            y_size=16,
            shared_reward=True,
            n_evaders=30,
            n_pursuers=8,
            obs_range=7,
            n_catch=2,
            freeze_evaders=False,
            tag_reward=0.01,
            catch_reward=5.0,
            urgency_reward=-0.1,
            surround=True,
            constraint_window=1.0,
        )

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


@hydra.main(version_base=None, config_path="config", config_name="rili")
def main(cfg: DictConfig):
    workspace = Workspace(cfg)
    if cfg.train:
        workspace.train()
    # workspace.eval()


if __name__ == "__main__":
    main()
