import random

import numpy as np


class ANOTHER_AGENT:
    def __init__(self, env, agent_type, random_prob=0.1):
        self.env = env
        self.random_prob = random_prob
        self.agent_type = agent_type

    def select_action(self, state):
        if random.random() < self.random_prob:
            return np.random.choice(5, size=1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]

        if self.agent_type == "fixed" or self.agent_type == "fixed_dynamics":
            fixed_landmark_no = self.env.world.fixed_landmark_no
            object_relpos = state[
                4 + 2 * (fixed_landmark_no) : 4 + 2 * (fixed_landmark_no + 1)
            ]
        elif self.agent_type == "following":
            object_relpos = state[-3:-1]
        else:
            raise ValueError("Unknown agent type")

        if abs(object_relpos[0]) > abs(object_relpos[1]):
            return 2 if object_relpos[0] > 0 else 1  # move_right or move_left
        else:
            return 4 if object_relpos[1] > 0 else 3  # move_up or move_down

    def just_select_action(self, state):
        return self.select_action(state)

    def change_env(self, env):
        self.env = env
