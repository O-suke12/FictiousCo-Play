import math
import random

import numpy as np


class ANOTHER_AGENT:
    def __init__(self, env, random_prob: float = 0.1):
        self.env = env
        self.random_prob = random_prob
        self.agent_type = None
        self.velocity = None
        self.speed = None

    def set_agent_type(self, agent_type):
        self.agent_type = agent_type
        if agent_type == "circle":
            sign = random.choice([-1, 1])
            self.speed = (math.pi / random.randint(5, 20)) * sign
            self.direction = np.array([(max(random.random(), 0.25)) * 2, 0])  # 初期方向
            self.rotation_matrix = np.array(
                [
                    [math.cos(self.speed), -math.sin(self.speed)],
                    [math.sin(self.speed), math.cos(self.speed)],
                ]
            )
            self.random_prob = 0.0
        else:
            self.random_prob = 0.1

    def select_action(self, state, t, end):
        if random.random() < self.random_prob:
            return np.random.choice(5, size=1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]

        if self.agent_type == "fixed" or self.agent_type == "fixed_dynamics":
            fixed_landmark_no = self.env.world.fixed_landmark_no
            object_relpos = state[
                4 + 2 * (fixed_landmark_no) : 4 + 2 * (fixed_landmark_no + 1)
            ]
        elif self.agent_type == "following":
            object_relpos = state[-4:-2]
        elif self.agent_type == "circle":
            self.velocity = state[0:2]
            x, y = state[2:4]
            return self.circle(x, y)
        else:
            raise ValueError("Unknown agent type")

        if abs(object_relpos[0]) > abs(object_relpos[1]):
            return 2 if object_relpos[0] > 0 else 1  # move_right or move_left
        else:
            return 4 if object_relpos[1] > 0 else 3  # move_up or move_down

    def just_select_action(self, state, t, end):
        return self.select_action(state, t, end)

    def change_env(self, env):
        self.env = env

    def circle(self, x, y):
        self.direction = np.dot(self.rotation_matrix, self.direction)
        delta_v = self.direction - self.velocity

        # 差が大きい方向に移動するアクションを選択
        if abs(delta_v[0]) > abs(delta_v[1]):
            action = 1 if delta_v[0] < 0 else 2
        else:
            action = 3 if delta_v[1] < 0 else 4

        return action
