import os
from datetime import datetime

import hydra
import numpy as np
import pytz
import torch
from omegaconf import DictConfig

import wandb

# from envs.mpe import simple_spread_v3
from envs.mpe_fixed_env import simple_spread_v3
from stock.PPO import PPO
from utils.recoder import VideoRecorder

"""
エージェントの種類
1.特定のポイントに向かうエージェント(同じポイントでないと学習できない)
2.特定のポイントに向かうエージェント(時間による移動付き)
3.ついていくエージェント
"""
# TODO 固定ポイント以外に動きをつける(時間でもらえる報酬を変える=>同じ場所に居続けてはいけない)
# TODO VAE
# TODO 論文を書き始める

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
    max_cycles=200,
    continuous_actions=False,
)
validation_env = simple_spread_v3.env(
    N=agent_num,
    LN=landmark_num,
    another_agent_type=another_agent_type,
    local_ratio=0.5,
    max_cycles=200,
    continuous_actions=False,
    render_mode="rgb_array",
)


root_dir = os.getcwd()
recoder = VideoRecorder(root_dir)

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
####### initialize environment hyperparameters ######

has_continuous_action_space = False  # continuous action space; else discrete

max_ep_len = 1000  # max timesteps in one episode
max_training_timesteps = int(
    3e6
)  # break training loop if timeteps > max_training_timesteps

print_freq = max_ep_len * 10  # print avg reward in the interval (in num timesteps)
log_freq = max_ep_len * 2  # log avg reward in the interval (in num timesteps)
save_model_freq = int(3e5)  # save model frequency (in num timesteps)

action_std = 0.6  # starting std for action distribution (Multivariate Normal)
action_std_decay_rate = (
    0.05  # linearly decay action_std (action_std = action_std - action_std_decay_rate)
)
min_action_std = (
    0.1  # minimum action_std (stop decay after action_std <= min_action_std)
)
action_std_decay_freq = int(2.5e5)  # action_std decay frequency (in num timesteps)
#####################################################

## Note : print/log frequencies should be > than max_ep_len

################ PPO hyperparameters ################
update_timestep = max_ep_len * 4  # update policy every n timesteps
K_epochs = 80  # update policy for K epochs in one PPO update

eps_clip = 0.2  # clip parameter for PPO
gamma = 0.99  # discount factor

lr_actor = 0.0003  # learning rate for actor network
lr_critic = 0.001  # learning rate for critic network

random_seed = 0  # set random seed if required (0 = no random seed)

state_dim = env.observation_space(env.possible_agents[0]).shape[0]

action_dim = env.action_space(env.possible_agents[0]).n


################################### Training ###################################
@hydra.main(version_base=None, config_path="config", config_name="ppo")
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

    print(
        "============================================================================================"
    )

    #####################################################

    print("training environment name : " + env_name)

    ################### checkpointing ###################
    run_num_pretrained = (
        0  #### change this to prevent overwriting weights in same env_name folder
    )

    #####################################################

    ############# print all hyperparameters #############
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_ep_len)
    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print(
        "printing average reward over episodes in last : "
        + str(print_freq)
        + " timesteps"
    )
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print(
        "--------------------------------------------------------------------------------------------"
    )

    print("Initializing a discrete action space policy")
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("PPO update frequency : " + str(update_timestep) + " timesteps")
    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)
    print(
        "--------------------------------------------------------------------------------------------"
    )
    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)
    if random_seed:
        print(
            "--------------------------------------------------------------------------------------------"
        )
        print("setting random seed to ", random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)
        np.random.seed(random_seed)
    #####################################################

    print(
        "============================================================================================"
    )

    ################# training procedure ################

    # initialize a PPO agent
    flex_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
        run=run,
        cfg=cfg,
    )
    another_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
        run=run,
        cfg=cfg,
    )
    agents = {}

    agents[env.possible_agents[0]] = another_agent
    agents[env.possible_agents[1]] = flex_agent

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

    # training loop
    while time_step <= max_training_timesteps:
        env.reset()
        current_ep_reward = 0

        for t in range(1, max_ep_len + 1):
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
                if time_step % update_timestep == 0:
                    another_agent.update()
                    flex_agent.update()

                # log in logging file
                if time_step % log_freq == 0:
                    # log average reward till last episode
                    log_avg_reward = log_running_reward / log_running_episodes
                    log_avg_reward = round(log_avg_reward, 4)

                    log_running_reward = 0
                    log_running_episodes = 0

                # printing average reward
                if time_step % print_freq == 0:
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
                if time_step % save_model_freq == 0:
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
        has_continuous_action_space,
        action_std,
    )
    another_agent = PPO(
        state_dim,
        action_dim,
        lr_actor,
        lr_critic,
        gamma,
        K_epochs,
        eps_clip,
        has_continuous_action_space,
        action_std,
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
