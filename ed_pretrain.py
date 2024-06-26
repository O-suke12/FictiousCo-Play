import os


def ed_pretrain(
    agents,
    env,
    flex_checkpoint_path,
    cfg,
):
    collect_another_agent = agents[env.possible_agents[0]]
    collect_flex_agent = agents[env.possible_agents[1]]

    trained_flex_path = flex_checkpoint_path.replace("lili", "ppo")
    if os.path.exists(trained_flex_path):
        collect_flex_agent.load(trained_flex_path)
    else:
        print("Checkpoint path does not exist.")
        exit(1)
    time_step = 0

    collect_agents = {}
    collect_agents[env.possible_agents[0]] = collect_another_agent
    collect_agents[env.possible_agents[1]] = collect_flex_agent

    while time_step <= cfg.encoder_decoder_training_timesteps:
        env.reset()
        for t in range(1, cfg.max_ep_len + 1):
            for agent in env.possible_agents:
                state, reward, done, truncated, info = env.last()
                action = collect_agents[agent].select_action(state)

                collect_agents[agent].buffer.rewards.append(reward)
                collect_agents[agent].buffer.is_terminals.append(done or truncated)
                time_step += 1
                if time_step % cfg.update_timestep == 0:
                    collect_flex_agent.ed_update()

                if done or truncated:
                    action = None
                env.step(action)
            if done or truncated:
                break
