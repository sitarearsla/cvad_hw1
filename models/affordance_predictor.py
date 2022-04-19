import torch
import torch.nn as nn
import torchvision.models as models
from numpy.lib.stride_tricks import as_strided
import numpy as np


class AffordancePredictor(nn.Module):
    """Afforance prediction network that takes images as input"""

    def __init__(self):
        super(AffordancePredictor, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet_connection = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
        )
        self.traffic_light_state = TaskBlock(n_in=512, n_out=2)
        self.traffic_light_distance = TaskBlock(n_in=512, n_out=1)
        self.lane_distance = TaskBlock(n_in=512, n_out=1, cond=True)
        self.route_angle = TaskBlock(n_in=512, n_out=1, cond=True)

    def forward(self, img, command):
        x = self.resnet18(img)
        x = self.resnet_connection(x)

        traffic_light_state = self.traffic_light_state(x)
        traffic_light_distance = self.traffic_light_distance(x)
        lane_distance = self.lane_distance(x, command)
        route_angle = self.route_angle(x, command)

        pred = {'tl_state': traffic_light_state,
                'tl_dist': traffic_light_distance,
                'lane_dist': lane_distance,
                'route_angle': route_angle}

        return pred


class TaskBlock(nn.Module):
    def __init__(self, n_in, n_out, cond=False):
        super(TaskBlock, self).__init__()
        self.cond = cond
        self.n_h = 64 if not cond else 64 * 4
        self.discrete = n_out > 1

        self.bn = nn.BatchNorm1d(self.n_h)
        self.dropout = nn.Dropout(0.5)
        self.lin_out = nn.Linear(self.n_h, n_out)

        self.core = nn.Sequential(
            nn.Linear(n_in, self.n_h),
            nn.ReLU(inplace=False),
        )

    def forward(self, x_in, d=None):
        x = self.core(x_in)

        if self.discrete:
            x = self.bn(x)
        x = self.dropout(x)

        # handle conditional affordances
        if self.cond:
            bool_vec = get_bool_vec(d, self.n_h).to(x.device)
            x *= bool_vec

        x = self.lin_out(x)
        return x


# helper functions from CAL paper

def tile_array(a, b0, b1):
    # create new 2D array
    r, c = a.shape
    rs, cs = a.strides
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0))
    return x.reshape(r * b0, c * b1)


def get_bool_vec(d, n_h):
    # first one hot encode to four possible classes
    d = d.detach().cpu()
    d = np.array(d).astype(np.int)
    t = np.zeros((len(d), 4))
    t[np.arange(len(d)), d] = 1
    # upscale to correct hidden_sz
    bool_vec = tile_array(t, 1, n_h // 4)
    return torch.Tensor(bool_vec)
