import os
import torch
import random
import os
import numpy as np
import warnings

warnings.filterwarnings('ignore')
from dataset import Dataset
from args import args
from process import Trainer
from model import VQVAE
import torch.utils.data as Data

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

def main():
    seed_everything(seed=2023)

    train_dataset = Dataset(device=args.device, mode='train', args=args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    args.data_shape = train_dataset.shape()
    test_dataset = Dataset(device=args.device, mode='test', args=args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print(args.data_shape)
    print('dataset initial ends')

    model = VQVAE(data_shape=args.data_shape, hidden_dim=args.d_model, n_embed=args.n_embed, block_num=args.block_num,
                    wave_length=args.wave_length)
    print('model initial ends')

    trainer = Trainer(args, model, train_loader, test_loader, verbose=True)
    print('trainer initial ends')

    trainer.train()


if __name__ == '__main__':
    main()

