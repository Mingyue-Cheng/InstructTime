import torch
import torch.utils.data as Data
from args import Train_data, Test_data

class Dataset(Data.Dataset):
    def __init__(self, device, mode, args):
        self.args = args
        if mode == 'train':
            self.ecgs_images = Train_data
        else:
            self.ecgs_images = Test_data
        self.device = device
        self.mode = mode

    def __len__(self):
        return len(self.ecgs_images)

    def __getitem__(self, item):
        ecg_img = torch.tensor(self.ecgs_images[item]).to(self.device)
        return ecg_img * 2.5

    def shape(self):
        return self.ecgs_images[0].shape
