import os

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import T_co


class AttitudeDataset(Dataset):

    def __init__(self, csv_path) -> None:
        super().__init__()
        self.csv = pd.read_csv(csv_path)

    def __getitem__(self, index) -> T_co:
        # print(index)
        # load the image
        return self.csv["Image_Path"][index], self.csv["Label"][index]

    def __len__(self):
        print("Lenght:", len(self.csv))
        return len(self.csv)


if __name__ == '__main__':

    print(torch.__version__)
    csv_path = "data.csv"
    os.chdir(os.path.join(os.getcwd(), "custom_dataset"))
    print(os.path.join(os.getcwd(), "data.csv"))

    attitude_dataset = AttitudeDataset(csv_path=csv_path)
    train_loader = DataLoader(dataset=attitude_dataset, shuffle=True, num_workers=4, batch_size=5)
    for i, data in enumerate(train_loader):
        img, lbl = data
        print("{0} {1} {2}:".format(i, img, lbl))
