import os

from envs.mpe_fixed_env import simple_spread_v3
from PPO import PPO
from utils.recoder import VideoRecorder


#################################### Testing ###################################
def test():
    print(
        "============================================================================================"
    )

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving
    root_dir = os.getcwd()
    recoder = VideoRecorder(root_dir)
    env_name = "simple_spread_v3"
    has_continuous_action_space = False
    max_ep_len = 500  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 5  # total num of testing episodes

    K_epochs = 80  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    #####################################################

    env = simple_spread_v3.env(
        N=2,
        local_ratio=0.5,
        max_cycles=100,
        continuous_actions=False,
        render_mode="rgb_array",
    )

    # state space dimension
    state_dim = env.observation_space(env.possible_agents[0]).shape[0]

    action_dim = env.action_space(env.possible_agents[0]).n

    # initialize a PPO agent
    fixed_agent = PPO(
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

    # preTrained weights directory

    random_seed = (
        0  #### set this to load a particular checkpoint trained on random seed
    )
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "models" + "/" + env_name + "/"
    fixed_checkpoint_path = directory + "fixed_PPO_{}_{}_{}.pth".format(
        env_name, random_seed, run_num_pretrained
    )
    flex_checkpoint_path = directory + "flex_PPO_{}_{}_{}.pth".format(
        env_name, random_seed, run_num_pretrained
    )

    print(
        "--------------------------------------------------------------------------------------------"
    )

    test_running_reward = 0
    agents = {}

    agents[env.possible_agents[0]] = fixed_agent
    agents[env.possible_agents[1]] = flex_agent
    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        env.reset()
        recoder.init()

        for t in range(1, max_ep_len + 1):
            for agent in env.possible_agents:
                state, reward, done, truncated, info = env.last()
                action = agents[agent].select_action(state)
                ep_reward += reward

                if render:
                    recoder.record(env)

                if done or truncated:
                    break
                env.step(action)
            if done or truncated:
                break

        # clear buffer
        flex_agent.buffer.clear()
        fixed_agent.buffer.clear()
        video_name = f"eval{ep}.mp4"
        recoder.save(video_name)

        test_running_reward += ep_reward
        print("Episode: {} \t\t Reward: {}".format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

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
    test()
