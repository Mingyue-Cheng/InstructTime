import torch
import random
import os
import pickle
import seaborn as sns
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import warnings
import scipy.stats

warnings.filterwarnings('ignore')
from collections import Counter
from dataset import Dataset
from args import args
import torch.utils.data as Data

from matplotlib.ticker import FuncFormatter

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def calc_ent(model, data_loader):
    id_counts = Counter() 
    with torch.no_grad(): 
        for batch in data_loader:
            for seq in batch:
                tensor = torch.Tensor(seq).unsqueeze(0).to(args.device)
                try:
                    _, _, id = model(tensor)
                    id_counts.update(id.squeeze().cpu().numpy())
                except Exception as e:
                    print(f"error in entroy: {e}")

    counts = np.array(list(id_counts.values()))
    probs = counts / counts.sum()
    entropy = scipy.stats.entropy(probs, base=2)

    return entropy, id_counts

def only_recon_loss(model, data_loader):
    mse = nn.MSELoss()
    total_recon_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in data_loader:
            seqs = batch
            out, _, _ = model(seqs)
            recon_loss = mse(out, seqs)

            total_recon_loss += recon_loss.item()
            total_batches += 1

    avg_recon_loss = total_recon_loss / total_batches
    return avg_recon_loss

def save_output_to_file(output, save_path):
    with open(save_path, "w") as file:
        file.write(output)

def case_analysis(model, data, save_path):
    for i in range(20):
        label, sample_data, _ = data[i]
        sample_data = sample_data * 2.5
        with torch.no_grad():
            sample_data_tensor = torch.Tensor(sample_data).unsqueeze(0).to(args.device)
            reconstructed, _, _ = model(sample_data_tensor)

        original_data = sample_data
        reconstructed_data = reconstructed.squeeze(0).cpu().numpy()
        dx_index = label.find("information.\n")
        label = label[dx_index + 13:]

        ecg_index = label.rfind("include(s) ")
        sleep_index = label.rfind("pattern is ")
        har_index = label.rfind("engaged in ")
        esr_index = label.rfind("state of ")

        if ecg_index != -1: 
            label = label[ecg_index + 11:]
        elif sleep_index != -1:
            label = label[sleep_index + 11:]
        elif esr_index != -1:
            label = label[esr_index + 9:]
        else:
            label = label[har_index + 11:]

        dim_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        num_dims = original_data.shape[1]
        fig, axes = plt.subplots(num_dims, 1, figsize=(12, num_dims * 2))
        
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        if num_dims == 1:
            plt.plot(original_data[:], label='Original Data') 
            plt.plot(reconstructed_data[:], label='Reconstructed Data') 
            plt.title(f"Data Comparison - {dim_name[0]} - label {label}")
            plt.legend(loc='upper left')
        else:
            for j in range(num_dims):
                axes[j].plot(original_data[:, j], label='Original Data')
                axes[j].plot(reconstructed_data[:, j], label='Reconstructed Data')
                axes[j].set_title(f"Data Comparison - {dim_name[j]} - label {label}")
                axes[j].legend(loc='upper left')
                
        def format_yaxis_label(value, pos):
            return '{:.1f}'.format(value)
        
        plt.gca().yaxis.set_major_formatter(FuncFormatter(format_yaxis_label))

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"dim_comparison_case{i}.svg"), format='svg')
        plt.close()
        
def plot_square_like_heatmap(id_counts, save_path, shape=(8, 16), font_size=24):
    heatmap_data = np.zeros(shape)

    for i, count in enumerate(id_counts.values()):
        if i >= np.prod(shape):
            break
        row = i // shape[1]
        column = i % shape[1]
        heatmap_data[row, column] = np.log1p(count)

    plt.figure(figsize=(20, 10))

    sns.set(font_scale=2.4)
    sns.heatmap(heatmap_data, cmap='Oranges', linecolor='white', linewidths=3, robust=True)
    
    plt.xticks([])
    plt.yticks([]) 

    plt.xlabel('Index Column', fontsize=font_size)
    plt.ylabel('Index Row', fontsize=font_size)
    plt.title('Heatmap of Frequency Distribution for Time Series Tokens', fontsize=font_size)

    plt.savefig(os.path.join(save_path, "square_like_heatmap.svg"), format='svg', dpi=300)
    plt.close()

def main():
    seed_everything(seed=2023)
    model_load_path = './test_ecg_64_128_40'
    output = []

    train_dataset = Dataset(device=args.device, mode='train', args=args)
    train_loader = Data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    args.data_shape = train_dataset.shape()
    test_dataset = Dataset(device=args.device, mode='test', args=args)
    test_loader = Data.DataLoader(test_dataset, batch_size=args.test_batch_size)
    print(args.data_shape)
    print('dataset initial ends')

    model = VQVAE(data_shape=args.data_shape, hidden_dim=args.d_model, n_embed=args.n_embed,
                    wave_length=args.wave_length)
    print('model initial ends')

    state_dict = torch.load(os.path.join(model_load_path, 'model.pkl'), map_location='cpu')
    model = model.to(args.device)
    model.load_state_dict(state_dict)
    model.eval()

    entropy, id_counts = calc_ent(model, train_loader)
    print('train:')
    print(entropy, len(id_counts))
    output.append('train:\n')
    output.append(f"{entropy} {len(id_counts)} {len(id_counts) / args.n_embed}\n")

    entropy, id_counts = calc_ent(model, test_loader)
    print('test:')
    print(entropy, len(id_counts))
    output.append('test:\n')
    output.append(f"{entropy} {len(id_counts)} {len(id_counts) / args.n_embed}\n")

    recon_loss = only_recon_loss(model, test_loader)
    print('recon loss(mse): {}\n'.format(recon_loss))
    output.append(f'recon loss(mse): {recon_loss}\n')

    plot_square_like_heatmap(id_counts, model_load_path)
    print("Images have been saved.")

    save_output_to_file(''.join(output), os.path.join(model_load_path, 'model_output.txt'))
    print("Texts have been saved.")

    file_path = '../ecg_no_big'
    test_path = os.path.join(file_path, 'samples_test.pkl')
    samples_test = []
    if os.path.isfile(test_path):
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)
    random_samples = random.sample(samples_test, 20)
    case_analysis(model, random_samples, model_load_path)

if __name__ == '__main__':
    main()