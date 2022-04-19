class RewardManager():
    """Computes and returns rewards based on states and actions."""

    def __init__(self):
        pass

    def get_reward(self, state, action):
        """Returns the reward as a dictionary. You can include different sub-rewards in the
        dictionary for plotting/logging purposes, but only the 'reward' key is used for the
        actual RL algorithm, which is generated from the sum of all other rewards."""
        # aiming for a safe driving experience, will reward accordingly
        # Your code here
        reward_dict = {'traffic_rules': self.reward_traffic_light_rules(state, action),
                       'collision': self.reward_collision(state),
                       'speed': self.reward_speed(state),
                       'command': self.reward_steer(state, action)}
        # Your code here
        reward = 0.0
        for val in reward_dict.values():
            reward += val
        reward_dict["reward"] = reward
        return reward_dict

    # positive reinforcement for true steering

    def reward_steer(self, state, action):
        if state['command'] == 0:  # left
            return 50 if action['steer'] < -1 else -50
        elif state['command'] == 1:  # right
            return 50 if action['steer'] > 1 else -50

    # negative reinforcements

    def reward_traffic_light_rules(self, state, action):
        if state['tl_state'] == 1 or state['hazard']:
            # penalize throttle and steer when red light or when hazard
            if action['brake'] > action['throttle'] or action['brake'] > action['throttle']:
                return -30
        return 0

    def reward_collision(self, state):
        if state['collision']:
            return -100
        return 0

    def reward_speed(self, state):
        if state["optimal_speed"] < state["speed"]:
            return -20
        return 0
