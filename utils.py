import os
import torch
import json
import random
from TStokenizer.model import TStokenizer

def get_fixed_order_choice(labels):
    shuffled_labels = labels[:]
    shuffled_labels = list(shuffled_labels)
    random.shuffle(shuffled_labels) 
    return shuffled_labels

def extract_all_information(text):
    diagnosis = stage = har = dev = whale = ""
    if "include(s)" in text:
        diagnosis = extract_from_text(text, "include(s) ")
    elif "pattern is" in text:
        stage = extract_from_text(text, "pattern is ")
    elif "engaged in" in text:
        har = extract_from_text(text, "engaged in ")
    elif "conditions:" in text:
        dev = extract_from_text(text, "conditions: ")
    elif "originates from" in text:
        whale = extract_from_text(text, "originates from ")
    return diagnosis, stage, har, dev, whale

def extract_from_text(text, keyword):
    index = text.find(keyword)
    if index != -1:
        return text[index + len(keyword):] 
    return ""

def load_params_from_json(json_file_path):
    with open(json_file_path, 'r') as file:
        params = json.load(file)
    return params

def load_TStokenizer(dir_path, data_shape, device):
    json_params_path = os.path.join(dir_path, "args.json")
    model_path = os.path.join(dir_path, "model.pkl")

    params = load_params_from_json(json_params_path)

    vqvae_model = TStokenizer(data_shape=data_shape, hidden_dim=params['d_model'], n_embed=params['n_embed'], wave_length=params['wave_length'])
    vqvae_model.load_state_dict(torch.load(model_path, map_location=device))
    vqvae_model.eval()
    
    return vqvae_model
