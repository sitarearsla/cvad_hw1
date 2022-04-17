import torch
from torch.utils.data import Dataset
import os
import cv2
import json
import numpy as np


class ExpertDataset(Dataset):
    """Dataset of RGB images, driving affordances and expert actions"""

    def __init__(self, data_root):
        self.data_root = data_root
        self.img_path = data_root + "/rgb"
        self.measurement_path = data_root + "/measurements"
        self.measurents_json=[json for json in os.listdir(self.measurement_path) if json.endswith('.json')]

    def __len__(self):
        return len(self.measurents_json)

    def __getitem__(self, index):
        """Return RGB images and measurements"""
        # Your code here
        index_str = str(index)
        for i in range(8-len(index_str)):
            index_str = "0" + index_str

        img_name = os.path.join(self.img_path,
                                index_str + ".png")
        #print(img_name)
        img = cv2.imread(img_name, cv2.IMREAD_COLOR)
        img = img.transpose(2, 0, 1)

        img = img.astype(np.float)
        img = torch.from_numpy(img).type(torch.FloatTensor)
        img = img / 255.
        json_item = self.measurement_path + "/" + self.measurents_json[index]
        with open(json_item) as json_file:
            measurement = json.load(json_file)
        for k, v in measurement.items():
            v = torch.from_numpy(np.asarray([v, ]))
            measurement[k] = v.float()

        measurement['rgb'] = img
        return measurement
