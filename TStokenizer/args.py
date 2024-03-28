import argparse
import os
import json
from datautils import load_all_data

parser = argparse.ArgumentParser()
# dataset and dataloader args
parser.add_argument('--save_path', type=str, default='./test')
parser.add_argument('--dataset', type=str, default='sleep', choices=['har', 'geo', 'sleep', 'dev', 'ecg', 'whale', 'ad', 'esr'])
parser.add_argument('--data_path', type=str, default=None)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--test_batch_size', type=int, default=64)

# model args
parser.add_argument('--d_model', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--eval_per_steps', type=int, default=300)
parser.add_argument('--enable_res_parameter', type=int, default=1)
parser.add_argument('--n_embed', type=int, default=2560)
parser.add_argument('--wave_length', type=int, default=25)
parser.add_argument('--mask_prob', type=float, default=0.2)
parser.add_argument('--pooling_type', type=str, default='mean', choices=['mean', 'max', 'last_token'])

# tcn args
parser.add_argument('--kernel_size', type=int, default=3)
parser.add_argument('--block_num', type=int, default=4)
parser.add_argument('--dilations', type=list, default=[1, 4])

# train args
parser.add_argument('--lr', type=float, default=0.0005)
parser.add_argument('--lr_decay_rate', type=float, default=0.99)
parser.add_argument('--lr_decay_steps', type=int, default=300)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--num_epoch', type=int, default=60)

args = parser.parse_args()
if args.data_path is None:
    if args.dataset == 'har':
        Train_data, Test_data = load_all_data('../har_no_big')
    elif args.dataset == 'geo':
        Train_data, Test_data = load_all_data('../ecg_no_big')
    elif args.dataset == 'dev':
        Train_data, Test_data = load_all_data('../device_no_big')
    elif args.dataset == 'whale':
        Train_data, Test_data = load_all_data('../whale_no_big')
    elif args.dataset == 'ad':
        Train_data, Test_data = load_all_data('../ad_no_big')
    elif args.dataset == 'esr':
        Train_data, Test_data = load_all_data('../esr_no_big')
    else:
        Train_data, Test_data = load_all_data('../eeg_no_big')
else:
    path = args.data_path
    if args.dataset == 'har':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'geo':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'dev':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'ecg':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'whale':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'ad':
        Train_data, Test_data = load_all_data(path)
    elif args.dataset == 'esr':
        Train_data, Test_data = load_all_data(path)
    else:
        Train_data, Test_data = load_all_data(path)
print('data loaded')

if args.save_path == 'None':
    path_str = 'D-' + str(args.d_model) + '_Model-' + args.model + '_Lr-' + str(args.lr) + '_Dataset-' + args.dataset + '/'
    args.save_path = path_str
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)

config_file = open(args.save_path + '/args.json', 'w')
tmp = args.__dict__
json.dump(tmp, config_file, indent=1)
print(args)
config_file.close()
