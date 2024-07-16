import os
from datetime import datetime

import hydra
import numpy as np
import pytz
import torch
from omegaconf import DictConfig

import wandb
from envs.mpe_fixed_env import simple_spread_v3
from stock.PPO import PPO
from utils.recoder import VideoRecorder

lili = False
env_name = "simple_spread_v3"
agent_num = 2
landmark_num = 3
agent_type = [None, "fixed", "fixed_dynamics", "following"]
another_agent_type = agent_type[0]
encoder_decoder_pretrain = False
env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=160,
    continuous_actions=False,
)
validation_env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=160,
    continuous_actions=False,
    render_mode="rgb_array",
)

if lili:
    agent_type = "lili"
    cuda_num = 0
else:
    agent_type = "ppo"
    cuda_num = 1

name = f"{another_agent_type}_{agent_type}_{agent_num}agent_{landmark_num}land"

state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).n
val_count = 0


root_dir = os.getcwd()
recoder = VideoRecorder(root_dir)

device = torch.device("cpu")

if torch.cuda.is_available():
    device = torch.device(f"cuda:{cuda_num}")
    torch.cuda.empty_cache()

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + "/" + env_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

another_checkpoint_path = directory + f"another_PPO_{name}.pth"
flex_checkpoint_path = directory + f"flex_PPO_{name}.pth"


################################### Training ###################################
@hydra.main(version_base=None, config_path="config", config_name="lili")
def train(cfg: DictConfig):
    japan_tz = pytz.timezone("Japan")
    now = datetime.now(japan_tz)

    if cfg.track:
        run_name = f"{name}_{now.strftime('%m_%d_%H:%M')}"
        run = wandb.init(
            project=cfg.project_name,
            sync_tensorboard=True,
            monitor_gym=True,
            name=run_name,
        )
    else:
        run = None

    ################# training procedure ################
    if lili:
        print(
            "============================================================================================"
        )
        print("Currently using LILI")
        print(
            "============================================================================================"
        )
        z_dim = 8
    else:
        print(
            "============================================================================================"
        )
        print("Currently using PPO")
        print(
            "============================================================================================"
        )
        z_dim = 0
    # initialize a PPO agent
    flex_agent = PPO(
        state_dim,
        action_dim,
        z_dim,
        device,
        cfg,
        run=run,
    )
    another_agent = PPO(
        state_dim,
        action_dim,
        0,
        device,
        cfg,
        run=run,
    )
    agents = {}

    agents[env.possible_agents[0]] = another_agent
    agents[env.possible_agents[1]] = flex_agent

    # encoder_decoder = EncoderDecoder(
    #     state_dim,
    #     action_dim,
    #     encoder_input_dim,
    #     z_dim,
    #     decoder_input_dim,
    #     decoder_output_dim,
    #     run=run,
    #     cfg=cfg,
    # )

    # track total training time
    start_time = now

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0
    flex_reward = 0
    another_reward = 0

    time_step = 0
    i_episode = 0

    # Collect data for Encoder-Decoder
    if lili and encoder_decoder_pretrain:
        collect_flex_agent = PPO(
            state_dim,
            action_dim,
            0,
            device,
            cfg,
            run=run,
        )
        collect_another_agent = PPO(
            state_dim,
            action_dim,
            0,
            device,
            cfg,
            run=run,
        )
        collect_agents = {}

        collect_agents[env.possible_agents[0]] = collect_another_agent
        collect_agents[env.possible_agents[1]] = collect_flex_agent
        print("Collecting data for Encoder-Decoder")
        trained_another_path = another_checkpoint_path.replace("lili", "ppo")
        trained_flex_path = flex_checkpoint_path.replace("lili", "ppo")
        if os.path.exists(trained_another_path):
            collect_another_agent.load(trained_another_path)
            collect_flex_agent.load(trained_flex_path)
        else:
            print("Checkpoint path does not exist.")
            exit(1)

        while time_step <= cfg.encoder_decoder_training_timesteps:
            env.reset()

            for t in range(1, cfg.max_ep_len + 1):
                for agent in env.possible_agents:
                    state, reward, done, truncated, info = env.last()
                    action = collect_agents[agent].select_action(state)

                    collect_agents[agent].buffer.rewards.append(reward)
                    collect_agents[agent].buffer.is_terminals.append(done)
                    time_step += 1
                    if time_step % cfg.update_timestep == 0:
                        PPO.update(collect_agents["flex_agent"])
                        another_agent.buffer.clear()
                        flex_agent.buffer.clear()

                    if done or truncated:
                        action = None
                        break
                    env.step(action)
                if done or truncated:
                    break

    # training loop
    if lili:
        while time_step <= cfg.max_training_timesteps:
            env.reset()
            current_ep_reward = 0

            for t in range(1, cfg.max_ep_len + 1):
                for agent in env.possible_agents:
                    state, reward, done, truncated, info = env.last()

                    if agent == "flex_agent":
                        flex_reward += reward
                        if len(agents[agent].buffer.states) > 1:
                            state_t_1 = (
                                agents[agent]
                                .buffer.states[-1]
                                .clone()
                                .detach()
                                .float()
                                .unsqueeze(0)
                            )
                            action_t_1 = (
                                agents[agent]
                                .buffer.actions[-1]
                                .clone()
                                .detach()
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )
                            reward_t_1 = (
                                (
                                    torch.tensor(agents[agent].buffer.rewards[-1])
                                    .float()
                                    .to(device)
                                )
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )
                            state_t = (
                                torch.tensor(state).float().unsqueeze(0).to(device)
                            )
                            tau = torch.cat(
                                (
                                    state_t_1,
                                    action_t_1,
                                    reward_t_1,
                                    state_t,
                                ),
                                dim=1,
                            )
                            agents[agent].buffer.tau.append(tau)
                            z = agents[agent].get_latent_strategies(tau)
                            z = z.cpu().detach().numpy()

                        else:
                            another_reward += reward
                            z = np.zeros((1, z_dim))

                        policy_input = np.concatenate((state, z[0]))
                        action = agents[agent].select_action(policy_input)

                    else:
                        action = agents[agent].select_action(state)

                    agents[agent].buffer.rewards.append(reward)
                    agents[agent].buffer.is_terminals.append(done)

                    time_step += 1
                    current_ep_reward += reward

                    # update PPO agent
                    if time_step % cfg.update_timestep == 0:
                        another_agent.update()
                        flex_agent.update()

                    # printing average reward
                    if time_step % cfg.print_freq == 0:
                        # print average reward till last episode
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)
                        flex_avg_reward = flex_reward / (print_running_episodes / 2)
                        flex_avg_reward = round(flex_avg_reward, 2)
                        another_avg_reward = another_reward / (
                            print_running_episodes / 2
                        )
                        another_avg_reward = round(another_avg_reward, 2)

                        print(
                            "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                                i_episode, time_step, print_avg_reward
                            )
                        )

                        print_running_reward = 0
                        print_running_episodes = 0

                        if cfg.track:
                            run.log(
                                {
                                    "average_reward": print_avg_reward,
                                    "flex_reward": flex_avg_reward,
                                    "another_reward": another_avg_reward,
                                }
                            )

                    # save model weights
                    if time_step % cfg.save_model_freq == 0:
                        print(
                            "--------------------------------------------------------------------------------------------"
                        )
                        flex_agent.save(flex_checkpoint_path)
                        another_agent.save(another_checkpoint_path)
                        print("model saved")
                        print(
                            "Elapsed Time  : ",
                            now - start_time,
                        )
                        print(
                            "--------------------------------------------------------------------------------------------"
                        )
                        validation(agents, cfg, run)

                    # break; if the episode is over
                    if done or truncated:
                        action = None
                        break
                    env.step(action)
                if done or truncated:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

        env.close()
    else:
        while time_step <= cfg.max_training_timesteps:
            env.reset()
            current_ep_reward = 0

            for t in range(1, cfg.max_ep_len + 1):
                for agent in env.possible_agents:
                    # select action with policy
                    state, reward, done, truncated, info = env.last()
                    action = agents[agent].select_action(state)

                    # saving reward and is_terminals
                    agents[agent].buffer.rewards.append(reward)
                    agents[agent].buffer.is_terminals.append(done)

                    time_step += 1
                    current_ep_reward += reward

                    # update PPO agent
                    if time_step % cfg.update_timestep == 0:
                        another_agent.update()
                        flex_agent.update()

                    # printing average reward
                    if time_step % cfg.print_freq == 0:
                        # print average reward till last episode
                        print_avg_reward = print_running_reward / print_running_episodes
                        print_avg_reward = round(print_avg_reward, 2)

                        print(
                            "Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(
                                i_episode, time_step, print_avg_reward
                            )
                        )

                        print_running_reward = 0
                        print_running_episodes = 0
                        if cfg.track:
                            run.log(
                                {
                                    "average_reward": print_avg_reward,
                                }
                            )

                    # save model weights
                    if time_step % cfg.save_model_freq == 0:
                        print(
                            "--------------------------------------------------------------------------------------------"
                        )
                        flex_agent.save(flex_checkpoint_path)
                        another_agent.save(another_checkpoint_path)
                        print("model saved")
                        print(
                            "Elapsed Time  : ",
                            now - start_time,
                        )
                        print(
                            "--------------------------------------------------------------------------------------------"
                        )
                        validation(agents, cfg, run)
                    # break; if the episode is over
                    if done or truncated:
                        action = None
                        break
                    env.step(action)
                if done or truncated:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            i_episode += 1

        env.close()

    # print total training time
    print(
        "============================================================================================"
    )
    end_time = now
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print(
        "============================================================================================"
    )


def validation(agents, cfg, run):
    test_running_reward = 0
    total_test_episodes = 1
    global val_count

    with torch.no_grad():
        for ep in range(1, total_test_episodes + 1):
            ep_reward = 0
            validation_env.reset()
            recoder.init()
            z_dim = 8
            z = np.zeros((1, z_dim))
            if lili:
                for t in range(1, cfg.max_ep_len + 1):
                    for agent in validation_env.possible_agents:
                        state, reward, done, truncated, info = validation_env.last()

                        if agent == "flex_agent":
                            if z.sum() != 0.0:
                                state_t = (
                                    torch.tensor(state).float().unsqueeze(0).to(device)
                                )
                                tau = torch.cat(
                                    (
                                        state_t_1,
                                        action_t_1,
                                        reward_t_1,
                                        state_t,
                                    ),
                                    dim=1,
                                )

                                z = agents[agent].get_latent_strategies(tau)
                                z = z.cpu().detach().numpy()

                            policy_input = np.concatenate((state, z[0]))
                            action = agents[agent].just_select_action(policy_input)

                            state_t_1 = (
                                torch.tensor(state).float().unsqueeze(0).to(device)
                            )
                            action_t_1 = (
                                torch.tensor(action)
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .to(device)
                            )
                            reward_t_1 = (
                                torch.tensor(reward)
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                                .to(device)
                            )

                        else:
                            action = agents[agent].just_select_action(state)

                        ep_reward += reward
                        if cfg.track:
                            recoder.record(validation_env)
                        # break; if the episode is over
                        if done or truncated:
                            action = None
                            break
                        validation_env.step(action)
                    if done or truncated:
                        break
            else:
                for t in range(1, cfg.max_ep_len + 1):
                    for agent in validation_env.possible_agents:
                        state, reward, done, truncated, info = validation_env.last()
                        action = agents[agent].just_select_action(state)

                        ep_reward += reward
                        if cfg.track:
                            recoder.record(validation_env)
                        # break; if the episode is over
                        if done or truncated:
                            action = None
                            break
                        validation_env.step(action)
                    if done or truncated:
                        break

            video_name = f"valid_{val_count}.mp4"
            recoder.save(video_name)
            if cfg.track:
                run.log(
                    {
                        "valid_average_reward": ep_reward,
                    }
                )
                run.log(
                    {"video": wandb.Video(f"{recoder.save_dir}/{video_name}", fps=24)}
                )

            test_running_reward += ep_reward
            print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))
            ep_reward = 0
            val_count += 1

    print(
        "============================================================================================"
    )

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print(
        "============================================================================================"
    )


def model_test():
    flex_agent = PPO(
        state_dim,
        action_dim,
        device,
        cfg,
    )
    another_agent = PPO(
        state_dim,
        action_dim,
        device,
        cfg,
    )
    flex_agent.load(flex_checkpoint_path)
    another_agent.load(another_checkpoint_path)
    print(f"loaded model from {flex_checkpoint_path} and {another_checkpoint_path}")
    agents = {}

    agents[validation_env.possible_agents[0]] = another_agent
    agents[validation_env.possible_agents[1]] = flex_agent

    test_running_reward = 0
    total_test_episodes = 3
    max_ep_len = 500

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        validation_env.reset()
        recoder.init()

        for t in range(1, max_ep_len + 1):
            for agent in validation_env.possible_agents:
                state, reward, done, truncated, info = validation_env.last()
                action = agents[agent].select_action(state)
                ep_reward += reward

                recoder.record(validation_env)

                if done or truncated:
                    break
                validation_env.step(action)
            if done or truncated:
                break

        for agent in validation_env.possible_agents:
            agents[agent].buffer.clear()
        agents[agent].buffer.clear()
        video_name = f"eval_{ep}.mp4"
        recoder.save(video_name)

        test_running_reward += ep_reward
        print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))
        ep_reward = 0

    print(
        "============================================================================================"
    )

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print(
        "============================================================================================"
    )


if __name__ == "__main__":
    train()
    # model_test()
