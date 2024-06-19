import os
from datetime import datetime

import hydra
import numpy as np
import pytz
import torch
from omegaconf import DictConfig

import wandb
from envs.mpe_fixed_env import simple_spread_v3
from latent import EncoderDecoder
from PPO import PPO
from utils.recoder import VideoRecorder

lili = False
env_name = "simple_spread_v3"
agent_num = 2
landmark_num = 3
agent_type = ["fixed", "fixed_dynamics", "following"]
another_agent_type = None
env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=120,
    continuous_actions=False,
)
validation_env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=120,
    continuous_actions=False,
    render_mode="rgb_array",
)

state_dim = env.observation_space(env.possible_agents[0]).shape[0]
action_dim = env.action_space(env.possible_agents[0]).n


root_dir = os.getcwd()
recoder = VideoRecorder(root_dir)

device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda:1")
    torch.cuda.empty_cache()

directory = "models"
if not os.path.exists(directory):
    os.makedirs(directory)

directory = directory + "/" + env_name + "/"
if not os.path.exists(directory):
    os.makedirs(directory)

another_checkpoint_path = directory + "another_PPO_{}_{}agent_{}land.pth".format(
    env_name, agent_num, landmark_num
)
flex_checkpoint_path = directory + "flex_PPO_{}_{}agent_{}land.pth".format(
    env_name, agent_num, landmark_num
)

################## Settings ########################

# max_ep_len = 1000  # max timesteps in one episode
# max_training_timesteps = int(
#     3e6
# )  # break training loop if timeteps > max_training_timesteps
# encoder_decoder_training_timesteps = int(1e2)

# print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
# log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
# save_model_freq = int(3e5)  # save model frequency (in num timesteps)


# ################ PPO hyperparameters ################

# update_timestep = max_ep_len * 4  # update policy every n timesteps
# encoder_decoder_update_timestep = 100
# K_epochs = 80  # update policy for K epochs in one PPO update

# eps_clip = 0.2  # clip parameter for PPO
# gamma = 0.99  # discount factor

# lr_actor = 0.0003  # learning rate for actor network
# lr_critic = 0.001  # learning rate for critic network

# random_seed = 0  # set random seed if required (0 = no random seed)


################################### Training ###################################
@hydra.main(version_base=None, config_path="config", config_name="lili")
def train(cfg: DictConfig):
    japan_tz = pytz.timezone("Japan")
    now = datetime.now(japan_tz)

    if cfg.track:
        run_name = f"{now.strftime('%m_%d_%H:%M')}"
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
        print("Currently using LILI")
        z_dim = 8
    else:
        print("Currently using PPO")
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

    encoder_input_dim = state_dim + 1 + 1 + state_dim

    decoder_input_dim = state_dim + 1 + z_dim
    decoder_output_dim = 1 + action_dim
    encoder_decoder = EncoderDecoder(
        state_dim,
        action_dim,
        encoder_input_dim,
        z_dim,
        decoder_input_dim,
        decoder_output_dim,
        run=run,
        cfg=cfg,
    )

    # track total training time
    start_time = now
    print("Started training at (GMT) : ", start_time)

    print(
        "============================================================================================"
    )

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0

    # VAE training loop
    # while time_step <= encoder_decoder_training_timesteps:
    #     vae_env.reset()

    #     for t in range(1, max_ep_len + 1):
    #         for agent in vae_env.possible_agents:
    #             # select action with policy
    #             state, reward, done, truncated, info = vae_env.last()
    #             action = agents[agent].select_action(state)

    #             # saving reward and is_terminals
    #             agents[agent].buffer.rewards.append(reward)
    #             agents[agent].buffer.is_terminals.append(done)

    #             time_step += 1

    #             # update PPO agent
    #             if time_step % encoder_decoder_update_timestep == 0:
    #                 encoder_decoder.update(agents["flex_agent"])
    #                 another_agent.buffer.clear()
    #                 flex_agent.buffer.clear()

    #             # break; if the episode is over
    #             if done or truncated:
    #                 action = None
    #                 break
    #             vae_env.step(action)
    #         if done or truncated:
    #             break

    # training loop

    if lili:
        while time_step <= cfg.max_training_timesteps:
            env.reset()
            current_ep_reward = 0

            for t in range(1, cfg.max_ep_len + 1):
                for agent in env.possible_agents:
                    state, reward, done, truncated, info = env.last()

                    if agent == "flex_agent":
                        if len(agents[agent].buffer.states) > 1:
                            state_t_minus1 = (
                                torch.tensor(agents[agent].buffer.states[-1])
                                .float()
                                .unsqueeze(0)
                            )
                            action_t_minus1 = (
                                torch.tensor(agents[agent].buffer.actions[-1])
                                .float()
                                .unsqueeze(0)
                                .unsqueeze(0)
                            )
                            reward_t_minus1 = (
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
                                    state_t_minus1,
                                    action_t_minus1,
                                    reward_t_minus1,
                                    state_t,
                                ),
                                dim=1,
                            )
                            z = encoder_decoder.get_latent_strategies(tau)
                            z = z.cpu().detach().numpy()

                        else:
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
                        encoder_decoder.update(agents["flex_agent"])
                        another_agent.update()
                        flex_agent.update()

                    # log in logging file
                    if time_step % cfg.log_freq == 0:
                        # log average reward till last episode
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)

                        log_running_reward = 0
                        log_running_episodes = 0

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
                        validation(agents, time_step)

                    # break; if the episode is over
                    if done or truncated:
                        action = None
                        break
                    env.step(action)
                if done or truncated:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

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

                    # log in logging file
                    if time_step % cfg.log_freq == 0:
                        # log average reward till last episode
                        log_avg_reward = log_running_reward / log_running_episodes
                        log_avg_reward = round(log_avg_reward, 4)

                        log_running_reward = 0
                        log_running_episodes = 0

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
                        validation(agents, time_step)

                    # break; if the episode is over
                    if done or truncated:
                        action = None
                        break
                    env.step(action)
                if done or truncated:
                    break

            print_running_reward += current_ep_reward
            print_running_episodes += 1

            log_running_reward += current_ep_reward
            log_running_episodes += 1

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


def validation(agents, time_step):
    test_running_reward = 0
    total_test_episodes = 1
    max_ep_len = 500

    with torch.no_grad():
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
            video_name = f"valid_{time_step}.mp4"
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


def model_test():
    flex_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
    )
    another_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
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
    model_test()
