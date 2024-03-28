import os
import pickle
import numpy as np

def save_datasets(data, path, file_name):
    if not os.path.exists(path):
        os.makedirs(path)
    np.save(os.path.join(path, file_name), data)

def load_datasets(path, file_name):
    return np.load(os.path.join(path, file_name))

def load_all_data(Path, use_saved_datasets=True):
    if use_saved_datasets:
        try:
            tokenizer_train = load_datasets(Path, 'train.npy')
            tokenizer_test = load_datasets(Path, 'test.npy')
            print(len(tokenizer_train), len(tokenizer_test))

            return tokenizer_train, tokenizer_test
        except IOError:
            print("Saved datasets not found. Processing raw data.")

    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train, tokenizer_train = [], []
    samples_test, tokenizer_test = [], []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for sample in samples_train:
        _, ecg, _ = sample
        tokenizer_train.append(ecg)
    for sample in samples_test:
        _, ecg, _ = sample
        tokenizer_test.append(ecg)

    tokenizer_train = np.array(tokenizer_train, dtype=np.float32)
    tokenizer_test = np.array(tokenizer_test, dtype=np.float32)

    save_datasets(tokenizer_train, Path, 'train.npy')
    save_datasets(tokenizer_test, Path, 'test.npy')

    return tokenizer_train, tokenizer_test

if __name__ == "__main__":
    load_all_data('./esr_data')
