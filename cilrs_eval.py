import os
import yaml
import numpy as np
from carla_env.env import Env
import torch


class Evaluator():
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.agent = self.load_agent()

    def load_agent(self):
        save_path = "cilrs_model.ckpt"
        model = torch.load(save_path)
        model.eval()
        if torch.cuda.is_available():
            model.cuda()
        return model

    def generate_action(self, rgb, command, speed):
        rgb = rgb.transpose(2, 0, 1)
        rgb = rgb.astype(np.float)
        rgb = torch.from_numpy(rgb).type(torch.FloatTensor)
        rgb = rgb / 255.
        rgb = rgb.unsqueeze(0)
        rgb = rgb.cuda()
        speed = np.array([speed]).astype(np.float)
        speed = torch.from_numpy(speed).type(torch.FloatTensor)
        speed = speed.unsqueeze(0)
        speed = speed.cuda()
        with torch.no_grad():
            branches = self.agent(rgb, speed)
            if command == 0:
                x = branches[1].data
            elif command == 1:
                x = branches[2].data
            elif command == 2:
                x = branches[3].data
            else:
                x = branches[0].data
        generated = x.detach().cpu()
        generated = np.array(generated).astype(np.float)
        throttle, brake, steer = generated[0]

        if throttle > brake:
            brake = 0
        elif brake > throttle:
            throttle = 0
        print(command, throttle, steer, brake)
        return throttle, brake, steer

    def take_step(self, state):
        rgb = state["rgb"]
        command = state["command"]
        speed = state["speed"]
        throttle, brake, steer = self.generate_action(rgb, command, speed)
        action = {
            "throttle": throttle,
            "brake": brake,
            "steer": steer
        }
        state, reward_dict, is_terminal = self.env.step(action)
        return state, is_terminal

    def evaluate(self, num_trials=100):
        terminal_histogram = {}
        for i in range(num_trials):
            state, _, is_terminal = self.env.reset()
            for i in range(5000):
                print('step'+str(i))
                if is_terminal:
                    break
                state, is_terminal = self.take_step(state)
            if not is_terminal:
                is_terminal = ["timeout"]
            terminal_histogram[is_terminal[0]] = (terminal_histogram.get(is_terminal[0], 0) + 1)
        print("Evaluation over. Listing termination causes:")
        for key, val in terminal_histogram.items():
            print(f"{key}: {val}/100")


def main():
    with open(os.path.join("configs", "cilrs.yaml"), "r") as f:
        config = yaml.full_load(f)

    with Env(config) as env:
        evaluator = Evaluator(env, config)
        evaluator.evaluate()


if __name__ == "__main__":
    main()
