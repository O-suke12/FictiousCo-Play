import random

import numpy as np


class ANOTHER_AGENT:
    def __init__(self, env, random_prob: float = 0.1):
        self.env = env
        self.random_prob = random_prob
        self.agent_type = None
        self.velocity = None
        self.speed = None
        self.x = None
        self.y = None
        self.region_x, self.region_y = None, None
        self.region_size = 0.5
        self.in_region = False
        self.grid_size = 1
        self.op_x = None
        self.op_y = None

    def set_agent_type(self, agent_type=None):
        # if agent_type is None:
        #     self.agent_type = random.choice(
        #         ["right_up", "left_down", "right_down", "left_up"]
        #     )
        # else:
        #     self.agent_type = agent_type
        # if self.agent_type == "right_up":
        #     self.region_x, self.region_y = 0.5, 0
        # elif self.agent_type == "left_down":
        #     self.region_x, self.region_y = 0, 0.5
        # elif self.agent_type == "right_down":
        #     self.region_x, self.region_y = 0.5, 0.5
        # elif self.agent_type == "left_up":
        #     self.region_x, self.region_y = 0, 0
        # else:
        #     assert False
        # pass
        if agent_type is None:
            self.agent_type = random.choice(
                ["Following without Overlapping", "Following with Overlapping"]
            )
        elif agent_type == "Following without Overlapping":
            self.agent_type = "Following without Overlapping"
        elif agent_type == "Following with Overlapping":
            self.agent_type = "Following with Overlapping"
        else:
            assert False

    def select_action(self, state, t, end):
        if self.agent_type == "Following with Overlapping":
            return self.following(state, t, end)

        elif self.agent_type == "Following without Overlapping":
            return self.escaping(state, t, end)

        else:
            assert False

    def following(self, state, t, end):
        self.x = state[2]
        self.y = state[1]

        self.op_x = state[-1]
        self.op_y = state[-2]

        x_diff = self.op_x - self.x
        y_diff = self.op_y - self.y

        if abs(x_diff) > abs(y_diff):
            if x_diff > 0:
                action = 4
            else:
                action = 2
        else:
            if (abs(x_diff) == 0) & (abs(y_diff) == 0):
                return 0
            if y_diff > 0:
                action = 1
            else:
                action = 3
        return action

    def escaping(self, state, t, end):
        self.x = state[2]
        self.y = state[1]

        self.op_x = state[-1]
        self.op_y = state[-2]

        x_diff = self.op_x - self.x
        y_diff = self.op_y - self.y

        if (abs(x_diff) == 0.25) & (abs(y_diff) == 0.25):
            action = 0
        elif ((x_diff == 0) & (abs(y_diff) == 0.25)) | (
            (abs(x_diff) == 0.25) & (y_diff == 0)
        ):
            if x_diff == 0:
                while True:
                    action = np.random.choice([2, 4], size=1, p=[0.5, 0.5])[0]
                    if self.valid_action(action):
                        break
            else:
                while True:
                    action = np.random.choice([1, 3], size=1, p=[0.5, 0.5])[0]
                    if self.valid_action(action):
                        break
        else:
            if abs(x_diff) > abs(y_diff):
                if x_diff > 0:
                    action = 4
                else:
                    action = 2
            else:
                if y_diff > 0:
                    action = 1
                else:
                    action = 3
        return action

    def _apply_action(self, action):
        if action == 3:
            return self.x, self.y - 0.25
        elif action == 1:
            return self.x, self.y + 0.25
        elif action == 2:
            return self.x - 0.25, self.y
        elif action == 4:
            return self.x + 0.25, self.y
        else:  # "noop"
            return self.x, self.y

    def valid_action(self, action):
        if action == 3:
            new_x, new_y = self.x, self.y - 0.25
        elif action == 1:
            new_x, new_y = self.x, self.y + 0.25
        elif action == 2:
            new_x, new_y = self.x - 0.25, self.y
        elif action == 4:
            new_x, new_y = self.x + 0.25, self.y
        else:  # "noop"
            new_x, new_y = self.x, self.y

        if new_x < 0 or new_x > 1 or new_y < 0 or new_y > 1:
            return False
        return True

    def _is_valid_position(self, x, y):
        return (
            self.region_x <= x <= self.region_x + self.region_size
            and self.region_y <= y <= self.region_y + self.region_size
        )

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


# def select_action(self, state, t, end):
#         # 重なるエージェント、重なりそうで重ならないエージェント
#         if random.random() < self.random_prob:
#             return np.random.choice(5, size=1, p=[0.2, 0.2, 0.2, 0.2, 0.2])[0]

#         self.x = state[2]
#         self.y = state[1]

#         if not self._is_valid_position(self.x, self.y):
#             x_diff = self.x - 0.5
#             y_diff = self.y - 0.5

#             if not self.region_x <= self.x <= self.region_x + self.region_size:
#                 if x_diff > 0:
#                     return 2
#                 else:
#                     return 4
#             elif not self.region_y <= self.y <= self.region_y + self.region_size:
#                 if y_diff > 0:
#                     return 3
#                 else:
#                     return 1

#         else:
#             while True:
#                 action = np.random.choice(
#                     5, size=1, p=[0.1, 0.225, 0.225, 0.225, 0.225]
#                 )[0]
#                 new_x, new_y = self._apply_action(action)
#                 if self._is_valid_position(new_x, new_y):
#                     return action
#                 else:
#                     continue
