import torch
import torch.nn as nn
import torchvision.models as models


class CILRS(nn.Module):
    """An imitation learning agent with a resnet backbone."""

    def __init__(self):
        super(CILRS, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")
        self.flatten = nn.Flatten()
        self.measurements = nn.Sequential(
            nn.Linear(1, 128),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
        )
        self.speed_prediction = nn.Sequential(
            nn.Linear(512, 256),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.Dropout2d(),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
        )
        self.join = nn.Sequential(
            nn.Linear(512 + 128, 512),
            # nn.Dropout2d(),
            nn.ReLU(inplace=True),
        )
        self.resnet_connection = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
        )

        self.branches = nn.ModuleList()
        for i in range(4):
            branch_head = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 256),
                nn.Dropout2d(),
                nn.ReLU(inplace=True),
                nn.Linear(256, 3),
            )
            self.branches.append(branch_head)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.1)

    def forward(self, img, speed):
        x = self.resnet18(img)
        x = self.resnet_connection(x)
        m = self.measurements(speed)
        j = self.join(torch.cat((x, m), 1))
        arr = []
        for branch in self.branches:
            result = branch(j)
            arr.append(result)
        s = self.speed_prediction(x)
        return arr + [s]
