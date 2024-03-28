import os, re
import ast,math, sys
import pywt, wfdb
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import pandas as pd
import torch
import pickle
import pandas as pd
import random
from collections import Counter
import neurokit2 as nk

def setup_seed(seed=2023):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def normalize(ecg):
    min_val = np.min(ecg)
    max_val = np.max(ecg)

    if (max_val - min_val) == 0:
        return ecg

    # 进行最大-最小归一化
    normalized_ecg = (ecg - min_val) / (max_val - min_val)

    return (normalized_ecg - 0.5) * 2

def denoising(data):
    # 初始化一个空矩阵来存储处理后的数据
    ecg_cleaned = np.zeros_like(data)

    # 循环处理每个通道
    for i in range(data.shape[1]):
        channel_data = data[:, i]
        ecg_cleaned[:, i] = nk.ecg_clean(channel_data, sampling_rate=500)

    return ecg_cleaned

def padding_varying_length(data):
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j, :][np.isnan(data[i, j, :])] = 0 
    return data

def pro_ecg(ecg):
    # filtered_data = denoising(ecg)
    normalize_data = normalize(ecg)

    return normalize_data

def trans_code(label_list):
    code_mapping = {
        733534002: 164909002,
        713427006: 59118001,
        284470004: 63593006,
        427172004: 17338001
    }

    # 使用列表推导式提高效率
    # 如果编码不在映射中，保持不变
    new_labelist = [code_mapping.get(code, code) for code in label_list]
    return new_labelist

def get_dict_ptb(Path='./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/scp_statements.csv', few=False):
    mapping_file = Path
    mapping_data = pd.read_csv(mapping_file)

    annotation_to_condition = {}
    for index, row in mapping_data.iterrows():
        if few:
            annotation_to_condition[row['diagnostic_class']] = index
        else:
            annotation_to_condition[row['description']] = index
            
    return annotation_to_condition

# def pro_text(comments, diction, only_label):
#     ann = comments[2]
#     snomed_ct_matches = re.findall(r'\d+', ann)

#     prefix = "Please provide a description of potential health problems, symptoms that the patient might be facing based on the provided information.\n"
#     new_snomed_ct_codes = ''
    
#     label_text = '['
#     for full_name in diction.values():
#         label_text += full_name + ','
#     if label_text.endswith(','):
#         label_text = label_text[:-1]
#     label_text += ']\nPlease select some symptoms from the table above for description.\n'

#     for snomed_ct_code in snomed_ct_matches:
#         label = diction.get(int(snomed_ct_code))
#         if label:
#             new_snomed_ct_codes += label + ','
#         else:
#             continue

#     if new_snomed_ct_codes.endswith(','):
#         new_snomed_ct_codes = new_snomed_ct_codes[:-1]

#     if new_snomed_ct_codes == '':
#         return ''
    
#     suffix = "This patient's symptoms include "

#     comments[2] = prefix + label_text + suffix + new_snomed_ct_codes 

#     colon_index = comments[0].find(":")
#     if colon_index != -1:
#         comments[0] = "Age:" + comments[0][colon_index + 1:]
#     else:
#         return ''

#     colon_index = comments[1].find(":")
#     if colon_index != -1:
#         comments[1] = "Gender:" + comments[1][colon_index + 1:]
#     else:
#         return ''

#     ann = comments[0]
#     age_matches = re.findall(r'\d+', ann)

#     for age_code in age_matches:
#         age_code_int = int(age_code)
#         if age_code_int <= 6:
#             ann = ann.replace(age_code, 'adolescence')
#         elif age_code_int <= 17:
#             ann = ann.replace(age_code, 'juvenile')
#         elif age_code_int <= 40:
#             ann = ann.replace(age_code, 'youths')
#         elif age_code_int <= 65:
#             ann = ann.replace(age_code, 'middle-age')
#         else:
#             ann = ann.replace(age_code, 'the elderly')
#     comments[0] = ann

#     if only_label:
#         text = comments[2]
#     else:
#         text = comments[0] + '\n' + comments[1] + '\n' + comments[2]

#     return text

def remove_before_colon(input_string):
    index = input_string.find(": ")
    if index != -1:
        return input_string[index+2:]
    else:
        return input_string

def get_dictionaries(path='C:/Users/ROG/Desktop/ConditionNames_SNOMED-CT.csv'):
    try:
        # 仅加载需要的列以减少内存占用
        mapping_data = pd.read_csv(path, usecols=['Snomed_CT', 'Full Name'])

        # 使用 pandas 的 to_dict 方法进行快速转换
        dict_snomed_to_name = mapping_data.set_index('Snomed_CT')['Full Name'].to_dict()
        dict_snomed_to_index = dict_snomed_to_index = mapping_data.reset_index().set_index('Snomed_CT')['index'].to_dict()

    except FileNotFoundError:
        raise FileNotFoundError(f"Unable to find the file at the specified path: {path}")
    except Exception as e:
        raise Exception(f"An error occurred while processing the file: {e}")

    return dict_snomed_to_name, dict_snomed_to_index

def process_12_lead_shot(data_folder='./data/12-lead/WFDBRecords/', only_label=False):
    diction1, diction2 = get_dictionaries('./data/12-lead.csv')
    label_frequency = {}  # Step 1: Create a label frequency dictionary
    num_classes = len(diction1)

    # Step 2: First pass to fill the frequency dictionary
    for file_out in os.scandir(data_folder):
        if file_out.is_dir():
            path_in = file_out.path
            for file_in in os.scandir(path_in):
                if file_in.is_dir():
                    data_path = file_in.path
                    for entry in os.scandir(data_path):
                        if entry.is_file() and entry.name.endswith('.hea'):
                            header = wfdb.rdheader(entry.path[:-4])
                            label = header.comments[2]
                            label = remove_before_colon(label)
                            label_frequency[label] = label_frequency.get(label, 0) + 1

    samples = []
    samples_fewshot = []
    samples_oneshot = []
    samples_zeroshot = []
    
    for file_out in os.scandir(data_folder):
        if file_out.is_dir():
            path_in = file_out.path
            for file_in in os.scandir(path_in):
                if file_in.is_dir():
                    data_path = file_in.path
                    for entry in os.scandir(data_path):
                        if entry.is_file() and entry.name.endswith('.hea'):
                            label_file = entry.path

                            record_name = label_file[:label_file.rfind('.')]
                            signals, _ = wfdb.rdsamp(record_name)
                            header = wfdb.rdheader(record_name)

                            if np.isnan(signals).any():
                                continue  # Skip NaN values

                            label = header.comments[2]
                            label = remove_before_colon(label)
                            label_list = label.split(',')
                            
                            label_indices = [diction2[int(snomed_ct_code)] for snomed_ct_code in label_list if int(snomed_ct_code) in diction2]
                            label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]

                            if not any(label_vector):
                                continue
                            
                            text = pro_text(header.comments, diction1, only_label)
                            ecg = pro_ecg(signals)

                            if label_frequency[label] > 20:
                                samples.append((text, ecg, label_vector))
                            elif label_frequency[label] <= 20 and label_frequency[label] > 10:
                                samples_fewshot.append((text, ecg, label_vector))
                            elif label_frequency[label] <= 10 and label_frequency[label] > 1:
                                samples_oneshot.append((text, ecg, label_vector))
                            elif label_frequency[label] == 1:
                                samples_zeroshot.append((text, ecg, label_vector))
    
    label_to_samples_map_fewshot = {}
    for sample in samples_fewshot:
        text, ecg, label_vector = sample
        label_index = label_vector.index(1)
        if label_index not in label_to_samples_map_fewshot:
            label_to_samples_map_fewshot[label_index] = []
        label_to_samples_map_fewshot[label_index].append(sample)

    test_fewshot = []
    for label_index, samples_list in label_to_samples_map_fewshot.items():
        # 从每个类别中选择七个样本并保留它们作为测试样本
        selected_samples = samples_list[-7:]
        test_fewshot.extend(selected_samples)

    train_fewshot = [sample for samples_list in label_to_samples_map_fewshot.values() for sample in samples_list[:-7]]
    np.random.shuffle(train_fewshot)

    label_to_samples_map_oneshot = {}
    for sample in samples_oneshot:
        text, ecg, label_vector = sample
        label_index = label_vector.index(1)
        if label_index not in label_to_samples_map_oneshot:
            label_to_samples_map_oneshot[label_index] = []
        label_to_samples_map_oneshot[label_index].append(sample)
    
    train_oneshot = []
    for label_index, samples_fewshot in label_to_samples_map_oneshot.items():
        # 从每个类别中选择一个样本并保留它作为训练样本
        selected_sample = samples_fewshot.pop()
        train_oneshot.append(selected_sample)

    test_oneshot = [sample for samples_list in label_to_samples_map_oneshot.values() for sample in samples_list]
    np.random.shuffle(train_oneshot)

    # Shuffle and split the samples as before
    index = [i for i in range(len(samples))]
    np.random.shuffle(index)
    split = int(0.9 * len(samples))
    samples_train = [samples[i] for i in index[:split]]
    samples_test = [samples[i] for i in index[split:]]

    samples_train = samples_train + train_oneshot + train_fewshot

    print(len(samples_train + samples_test))
    print(len(test_fewshot))
    print(len(test_oneshot))
    print(len(samples_zeroshot))

    # Save the filtered samples as before
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)
    with open('test_fewshot.pkl', 'wb') as file:
        pickle.dump(test_fewshot, file)
    with open('test_oneshot.pkl', 'wb') as file:
        pickle.dump(test_oneshot, file)
    with open('samples_test_zeroshot.pkl', 'wb') as file:
        pickle.dump(samples_zeroshot, file)

    return samples_train, samples_test

def process_12_lead(data_folder='./data/12-lead/WFDBRecords/', only_label=False):
    diction1, diction2 = get_dictionaries('./data/12-lead.csv')
    num_classes = len(diction1)
    samples = []
    
    for file_out in os.scandir(data_folder):
        if file_out.is_dir():
            path_in = file_out.path
            for file_in in os.scandir(path_in):
                if file_in.is_dir():
                    data_path = file_in.path
                    for entry in os.scandir(data_path):
                        if entry.is_file() and entry.name.endswith('.hea'):
                            label_file = entry.path

                            record_name = label_file[:label_file.rfind('.')]
                            signals, _ = wfdb.rdsamp(record_name)
                            header = wfdb.rdheader(record_name)

                            if np.isnan(signals).any():
                                continue  # Skip NaN values

                            label = header.comments[2]
                            label = remove_before_colon(label)
                            label_list = label.split(',')
                            
                            label_indices = [diction2[int(snomed_ct_code)] for snomed_ct_code in label_list if int(snomed_ct_code) in diction2]
                            
                            label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                            if not any(label_vector):
                                continue

                            text = pro_text(header.comments, diction1, only_label)
                            ecg = pro_ecg(signals)

                            if label_indices:
                                samples.append((text, ecg, label_vector))

    # Shuffle and split the samples as before
    index = [i for i in range(len(samples))]
    np.random.shuffle(index)
    split = int(0.9 * len(samples))
    samples_train = [samples[i] for i in index[:split]]
    samples_test = [samples[i] for i in index[split:]]

    print(len(samples_train + samples_test))

    # Save the filtered samples as before
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def pro_pd(label, label_dict, few=False):
    prefix = "Please provide a description of potential health problems, symptoms, and conditions that the patient might be facing based on the provided information.\nThis patient's symptoms include "
    if few:
        if not label['diagnostic_superclass']:
            return '', 0
        prefix += label['diagnostic_superclass'][0]
    else:
        if not label['description']:
            return '', 0
        prefix += label['description'][0]

    gender = 'Gender: '
    if label['sex'] == 1:
        gender += 'male\n'
    else: 
        gender += 'female\n'

    age = 'Age: '
    if label['age'] <= 6:
        age += 'adolescence\n'
    elif label['age'] <= 17:
        age += 'juvenile\n'
    elif label['age'] <= 40:
        age += 'youths\n'
    elif label['age'] <= 65:
        age += 'middle-age\n'
    else:
        age += 'the elderly\n'

    height = 'Height: '
    if label['height'] is not None:
        height += str(label['height'])
        height += ' cm\n'
    else:
        height += 'unknown\n'

    weight = 'Weight: '
    if label['weight'] is not None:
        weight += str(label['weight'])
        weight += ' kg\n'
    else:
        weight += 'unknown\n'

    text = gender + age + height + weight + prefix

    if few:
        if label['diagnostic_superclass'][0] == 'NORM':
            vector = 0
        elif label['diagnostic_superclass'][0] == 'STTC':
            vector = 1
        elif label['diagnostic_superclass'][0] == 'MI':
            vector = 2
        elif label['diagnostic_superclass'][0] == 'CD':
            vector = 3
        elif label['diagnostic_superclass'][0] == 'HYP':
            vector = 4
    else:
        vector = label_dict[label['description'][0]]

    return text, vector

def process_ptbxl(data_folder='./data/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/', few=True):
    def aggregate_description(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].description)
        return list(set(tmp))
    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    samples = []
    diction = get_dict_ptb(few=few)

    # Load and convert annotation data
    Y = pd.read_csv(data_folder + 'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # data = [wfdb.rdsamp(data_folder + f) for f in Y.filename_hr]
    
    # data = np.array([signal for signal, meta in data]).astype(np.float32)

    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(data_folder + 'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    # Apply diagnostic superclass
    Y['description'] = Y.scp_codes.apply(aggregate_description)
    Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

    for _, item in Y.iterrows():
        signals, _ = wfdb.rdsamp(data_folder + item.filename_hr)
        # header = wfdb.rdheader(data_folder + f)

        if np.isnan(signals).any():
            continue  # Skip NaN values

        text, vector = pro_pd(item, diction, few=few)
        ecg = pro_ecg(signals)

        if ecg.shape == (5000, 12) and text != '':
            samples.append((text, ecg, vector))

    # Shuffle and split the samples as before
    index = [i for i in range(len(samples))]
    np.random.shuffle(index)
    split = int(0.9 * len(samples))
    samples_train = [samples[i] for i in index[:split]]
    samples_test = [samples[i] for i in index[split:]]

    # Save the filtered samples as before
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_Georgia(data_folder='./data/georgia/', only_label=False):
    if only_label:
        diction1, diction2 = get_dictionaries('./data/geo-score.csv')
    else:
        diction1, diction2 = get_dictionaries('./data/essy.csv')
    num_classes = len(diction1)
    print(num_classes)
    samples = []

    for file_in in os.scandir(data_folder):
        if file_in.is_dir():
            data_path = file_in.path
            for entry in os.scandir(data_path):
                if entry.is_file() and entry.name.endswith('.hea'):
                    label_file = entry.path
                    header = wfdb.rdheader(label_file[:label_file.rfind('.')])
                    label = remove_before_colon(header.comments[2])

                    signals, _ = wfdb.rdsamp(label_file[:label_file.rfind('.')])
                    if np.isnan(signals).any():
                        continue  # Skip NaN values

                    label_list = label.split(',')
                    label_list = [int(item) for item in label_list]
                    # print(label_list)

                    # if only_label:
                    #     label_list = trans_code(label_list)
                    label_indices = [diction2[code] for code in label_list if code in diction2]
                    # print(diction2)
                    label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                    # print(label_indices)
                    if not any(label_vector):
                        continue

                    text = pro_text(header.comments, diction1)
                    # print(text)
                    # break
                    ecg = pro_ecg(signals)
                    if ecg.shape == (5000, 12):
                        samples.append((text, ecg, label_vector))

    np.random.shuffle(samples)
    split = int(0.9 * len(samples))
    samples_train = samples[:split]
    samples_test = samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_ecg(data_folder1='./data/georgia', data_folder2='./data/cpsc_2018', only_label=False):
    if only_label:
        diction1, diction2 = get_dictionaries('./data/geo-score.csv')
    else:
        diction1, diction2 = get_dictionaries('./data/essy.csv')
    num_classes = len(diction1)
    samples = []

    for file_in in os.scandir(data_folder1):
        if file_in.is_dir():
            data_path = file_in.path
            for entry in os.scandir(data_path):
                if entry.is_file() and entry.name.endswith('.hea'):
                    label_file = entry.path
                    header = wfdb.rdheader(label_file[:label_file.rfind('.')])
                    label = remove_before_colon(header.comments[2])

                    signals, _ = wfdb.rdsamp(label_file[:label_file.rfind('.')])
                    if np.isnan(signals).any():
                        continue  # Skip NaN values

                    label_list = label.split(',')
                    label_list = [int(item) for item in label_list]
                    # print(label_list)

                    # if only_label:
                    #     label_list = trans_code(label_list)
                    label_indices = [diction2[code] for code in label_list if code in diction2]
                    # print(diction2)
                    label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                    # print(label_indices)
                    if not any(label_vector):
                        continue

                    text = pro_text(header.comments, diction1)
                    # print(text)
                    # break
                    ecg = pro_ecg(signals)
                    if ecg.shape == (5000, 12):
                        samples.append((text, ecg, label_vector))

    for file_out in os.scandir(data_folder2):
        if file_out.is_dir():
            path_in = file_out.path
            for entry in os.scandir(path_in):
                if entry.is_file() and entry.name.endswith('.hea'):
                    label_file = entry.path

                    record_name = label_file[:label_file.rfind('.')]
                    signals, _ = wfdb.rdsamp(record_name)
                    header = wfdb.rdheader(record_name)

                    if np.isnan(signals).any():
                        continue  # Skip NaN values

                    label = header.comments[2]
                    label = remove_before_colon(label)
                    label_list = label.split(',')
                    label_list = [int(item) for item in label_list]
                    label_indices = [diction2[snomed_ct_code] for snomed_ct_code in label_list if snomed_ct_code in diction2]

                    label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                    if not any(label_vector):
                        continue

                    text = pro_text(header.comments, diction1)
                    ecg = pro_ecg(signals)

                    if ecg.shape == (5000, 12):
                        samples.append((text, ecg, label_vector))

    np.random.shuffle(samples)
    split = int(0.9 * len(samples))
    samples_train = samples[:split]
    samples_test = samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_ptb_xl_mul(data_folder='./data/ptb-xl-multi/', only_label=False):
    if only_label:
        diction1, diction2 = get_dictionaries('./data/ptbxl-score.csv')
    else:
        diction1, diction2 = get_dictionaries('./data/ptbxl.csv')
    num_classes = len(diction1)
    samples = []

    for file_out in os.scandir(data_folder):
        if file_out.is_dir():
            path_in = file_out.path
            for entry in os.scandir(path_in):
                if entry.is_file() and entry.name.endswith('.hea'):
                    record_name = entry.path[:-4]
                    header = wfdb.rdheader(record_name)
                    label = remove_before_colon(header.comments[2])

                    signals, _ = wfdb.rdsamp(record_name)
                    if np.isnan(signals).any():
                        continue

                    label_list = label.split(',')
                    label_indices = [diction2[int(code)] for code in label_list if int(code) in diction2]
                    label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                    if not any(label_vector):
                        continue

                    text = pro_text(header.comments, diction1)
                    ecg = pro_ecg(signals)
                    if ecg.shape == (5000, 12):
                        samples.append((text, ecg, label_vector))

    np.random.shuffle(samples)
    split = int(0.9 * len(samples))
    samples_train = samples[:split]
    samples_test = samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_cpsc_mul(data_folder='./data/cpsc_2018/', only_label=False):
    if only_label:
        diction1, diction2 = get_dictionaries('./data/cpsc-score.csv')
    else:
        diction1, diction2 = get_dictionaries('./data/essy.csv')
    num_classes = len(diction1)
    samples = []

    for file_out in os.scandir(data_folder):
        if file_out.is_dir():
            path_in = file_out.path
            for entry in os.scandir(path_in):
                if entry.is_file() and entry.name.endswith('.hea'):
                    label_file = entry.path

                    record_name = label_file[:label_file.rfind('.')]
                    signals, _ = wfdb.rdsamp(record_name)
                    header = wfdb.rdheader(record_name)

                    if np.isnan(signals).any():
                        continue  # Skip NaN values

                    label = header.comments[2]
                    label = remove_before_colon(label)
                    label_list = label.split(',')
                    label_list = [int(item) for item in label_list]
                    label_indices = [diction2[snomed_ct_code] for snomed_ct_code in label_list if snomed_ct_code in diction2]

                    label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
                    if not any(label_vector):
                        continue

                    text = pro_text(header.comments, diction1)
                    ecg = pro_ecg(signals)

                    if ecg.shape == (5000, 12):
                        samples.append((text, ecg, label_vector))

    # Shuffle and split the samples as before
    index = [i for i in range(len(samples))]
    np.random.shuffle(index)
    split = int(0.9 * len(samples))
    samples_train = [samples[i] for i in index[:split]]
    samples_test = [samples[i] for i in index[split:]]

    # Save the filtered samples as before
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)
    with open('zeroshot.pkl', 'wb') as file:
        pickle.dump(samples, file)

    print(len(samples))

    return samples_train, samples_test

def pro_text(comments, diction, only_label=False):
    ann = comments[2]
    snomed_ct_matches = re.findall(r'\d+', ann)

    prefix = "Describe the potential health issue(s) and associated symptom(s) the patient may be experiencing based on the provided information.\nThe symptom(s) exhibited by this patient include(s) "
    new_snomed_ct_codes = ''

    for snomed_ct_code in snomed_ct_matches:
        label = diction.get(int(snomed_ct_code))
        if label:
            new_snomed_ct_codes += label + ','
        else:
            continue

    if new_snomed_ct_codes.endswith(','):
        new_snomed_ct_codes = new_snomed_ct_codes[:-1]

    if new_snomed_ct_codes == '':
        return ''

    comments[2] = prefix + new_snomed_ct_codes 

    colon_index = comments[0].find(":")
    if colon_index != -1:
        comments[0] = "Age:" + comments[0][colon_index + 1:]
    else:
        return ''

    colon_index = comments[1].find(":")
    if colon_index != -1:
        comments[1] = "Gender:" + comments[1][colon_index + 1:]
    else:
        return ''

    ann = comments[0]
    age_matches = re.findall(r'\d+', ann)

    for age_code in age_matches:
        age_code_int = int(age_code)
        if age_code_int <= 6:
            ann = ann.replace(age_code, 'adolescence')
        elif age_code_int <= 17:
            ann = ann.replace(age_code, 'juvenile')
        elif age_code_int <= 40:
            ann = ann.replace(age_code, 'youths')
        elif age_code_int <= 65:
            ann = ann.replace(age_code, 'middle-age')
        else:
            ann = ann.replace(age_code, 'the elderly')
    comments[0] = ann

    if only_label:
        text = comments[2]
    else:
        text = comments[0] + '\n' + comments[1] + '\n' + comments[2]

    return text

def process_ecg_data(data_folder, dict_path, multi_folder=False, only_label=False):
    diction1, diction2 = get_dictionaries(dict_path)
    label_frequency = {}
    num_classes = len(diction1)
    samples = []

    def process_entry(entry):
        if entry.is_file() and entry.name.endswith('.hea'):
            record_name = entry.path[:-4]
            header = wfdb.rdheader(record_name)
            label = remove_before_colon(header.comments[2])
            label_frequency[label] = label_frequency.get(label, 0) + 1

            signals, _ = wfdb.rdsamp(record_name)
            if np.isnan(signals).any():
                return None

            label_list = label.split(',')
            label_indices = [diction2[int(code)] for code in label_list if int(code) in diction2]
            label_vector = [1 if i in label_indices else 0 for i in range(num_classes)]
            if not any(label_vector):
                return None

            text = pro_text(header.comments, diction1, only_label)
            ecg = pro_ecg(signals)
            if ecg.shape == (5000, 12):
                return (text, ecg.transpose(-1, 0), label_vector)
            else:
                return None

    for file_out in os.scandir(data_folder):
        if multi_folder and file_out.is_dir():
            path_in = file_out.path
            for entry in os.scandir(path_in):
                result = process_entry(entry)
                if result:
                    samples.append(result)
        elif not multi_folder:
            result = process_entry(file_out)
            if result:
                samples.append(result)

    np.random.shuffle(samples)
    split = int(0.9 * len(samples))
    samples_train = samples[:split]
    samples_test = samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_eeg(data_folder='./data/sleep-edf-database-1.0.0'):
    samples = []

    labels = np.load(os.path.join(data_folder, 'label.npy'))
    ecgs = np.load(os.path.join(data_folder, 'data.npy'))

    # 遍历数组并规范化
    prefix = "Select a previously mentioned sleep pattern and report on the person's sleep using the provided information.\nThe person's sleep pattern is "
    midfix = 'The sleep patterns include waking up, rapid eye movement sleep, and sleep stages one through four, as well as periods of movement and unidentified stages.\n'
    for ecg, label in zip(ecgs, labels):
        # 检查是否为 NaN 值
        if np.isnan(ecg).any():
            continue  # 丢弃包含 NaN 值的 ECG 数据
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(ecg)  # 使用您的规范化函数
        # print(label)
        if int(label) == 0:
            text = 'waking up'
        elif int(label) == 1:
            text = 'rapid eye movement sleep'
        elif int(label) == 2:
            text = 'sleep stage one'    
        elif int(label) == 3:
            text = 'sleep stage two'
        elif int(label) == 4:
            text = 'sleep stage three'
        elif int(label) == 5:
            text = 'sleep stage four'
        elif int(label) == 6:
            text = 'period of movement'
        elif int(label) == 7:
            text = 'unidentified stage'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        
        samples.append((text, ecg, label_vector))

    # samples = samples[:10000]
    np.random.shuffle(samples)
    split = int(0.9 * len(samples))
    samples_train = samples[:split]
    samples_test = samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_har(data_folder='./data/HAR'):
    normalized_samples_train = []
    normalized_samples_test = []

    data_val = torch.load('./data/HAR/val.pt')
    data_train = torch.load('./data/HAR/train.pt')
    data_test = torch.load('./data/HAR/test.pt')

    samples_train = data_train['samples'].numpy()
    labels_train = data_train['labels'].numpy()
    samples_test = data_test['samples'].numpy()
    labels_test = data_test['labels'].numpy()
    samples_val = data_val['samples'].numpy()
    labels_val = data_val['labels'].numpy()

    samples = np.concatenate([samples_train, samples_val], axis=0)
    labels = np.concatenate([labels_train, labels_val], axis=0)

    # 遍历数组并规范化
    prefix = "Please choose one activity from the previously mentioned six options and analyze the individual's physical activity based on the provided information.\nThe individual is currently engaged in "
    midfix = 'Physical activities such as walking, ascending stairs, descending stairs, sitting, standing, and lying down are recorded using mobile phone sensors.\n'
    
    for ecg, label in zip(samples, labels):
        # 检查是否为 NaN 值
        if np.isnan(ecg).any():
            continue  # 丢弃包含 NaN 值的 ECG 数据
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(ecg.astype(np.float32))  # 使用您的规范化函数
        ecg = ecg.transpose()

        if int(label) == 0:
            text = 'walking'
        elif int(label) == 1:
            text = 'ascending stairs'
        elif int(label) == 2:
            text = 'descending stairs'    
        elif int(label) == 3:
            text = 'sitting'
        elif int(label) == 4:
            text = 'standing'
        elif int(label) == 5:
            text = 'lying down'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (128, 9):
            normalized_samples_train.append((text, ecg, label_vector))
    
    for ecg, label in zip(samples_test, labels_test):
        # 检查是否为 NaN 值
        if np.isnan(ecg).any():
            continue  # 丢弃包含 NaN 值的 ECG 数据
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(ecg.astype(np.float32))  # 使用您的规范化函数
        ecg = ecg.transpose()

        if int(label) == 0:
            text = 'walking'
        elif int(label) == 1:
            text = 'ascending stairs'
        elif int(label) == 2:
            text = 'descending stairs'    
        elif int(label) == 3:
            text = 'sitting'
        elif int(label) == 4:
            text = 'standing'
        elif int(label) == 5:
            text = 'lying down'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (128, 9):
            normalized_samples_test.append((text, ecg, label_vector))

    # np.random.shuffle(normalized_samples)
    # split = int(0.8 * len(normalized_samples))
    # samples_train = normalized_samples[:split]
    # samples_test = normalized_samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def padding_varying_length(data):
    data[np.isnan(data)] = 0
    return data

def process_ad(data_folder='./datas/AD_data'):
    normalized_samples_train = []
    normalized_samples_test = []

    data_train = torch.load('./datas/AD_data/train.pt')
    data_test = torch.load('./datas/AD_data/test.pt')

    samples_train = data_train['samples'].numpy()
    labels_train = data_train['labels'].numpy()
    samples_test = data_test['samples'].numpy()
    labels_test = data_test['labels'].numpy()

    # 遍历数组并规范化
    prefix = "Please select one activity from the previously mentioned ten digits and analyze the individual's handwriting based on the provided information.\nThe person is currently writing the digit "
    midfix = 'Physical activities that specifically involve using a pen to write digits, which range from one to ten.\n'
    
    for ecg, label in zip(samples_train, labels_train):
        # # 检查是否为 NaN 值
        # padding_varying_length(ecg)
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(ecg.astype(np.float32))  # 使用您的规范化函数

        if int(label) == 0:
            text = 'zero'
        elif int(label) == 1:
            text = 'one'
        elif int(label) == 2:
            text = 'two'    
        elif int(label) == 3:
            text = 'three'
        elif int(label) == 4:
            text = 'four'
        elif int(label) == 5:
            text = 'five'
        elif int(label) == 6:
            text = 'six'    
        elif int(label) == 7:
            text = 'seven'
        elif int(label) == 8:
            text = 'eight'
        elif int(label) == 9:
            text = 'nine'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (93, 13):
            normalized_samples_train.append((text, ecg, label_vector))
    
    for ecg, label in zip(samples_test, labels_test):
        # # 检查是否为 NaN 值
        # padding_varying_length(ecg)
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(ecg.astype(np.float32))  # 使用您的规范化函数

        if int(label) == 0:
            text = 'zero'
        elif int(label) == 1:
            text = 'one'
        elif int(label) == 2:
            text = 'two'    
        elif int(label) == 3:
            text = 'three'
        elif int(label) == 4:
            text = 'four'
        elif int(label) == 5:
            text = 'five'
        elif int(label) == 6:
            text = 'six'    
        elif int(label) == 7:
            text = 'seven'
        elif int(label) == 8:
            text = 'eight'
        elif int(label) == 9:
            text = 'nine'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (93, 13):
            normalized_samples_test.append((text, ecg, label_vector))

    # np.random.shuffle(normalized_samples)
    # split = int(0.8 * len(normalized_samples))
    # samples_train = normalized_samples[:split]
    # samples_test = normalized_samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def process_esr(data_folder='./data/HAR'):
    normalized_samples_train = []
    normalized_samples_test = []

    data_val = torch.load('./datas/esr/val.pt')
    data_train = torch.load('./datas/esr/train.pt')
    data_test = torch.load('./datas/esr/test.pt')

    samples_train = data_train['samples'].numpy()
    labels_train = data_train['labels'].numpy()
    samples_test = data_test['samples'].numpy()
    labels_test = data_test['labels'].numpy()
    samples_val = data_val['samples'].numpy()
    labels_val = data_val['labels'].numpy()

    samples = np.concatenate([samples_train, samples_val], axis=0)
    labels = np.concatenate([labels_train, labels_val], axis=0)

    # 遍历数组并规范化
    prefix = "Please choose one of the two previously mentioned labels and analyze the individual's condition for a possible epilepsy diagnosis based on the provided information.\nAt this moment, the individual is existing in a particular state of "
    midfix = "In the clinical evaluation, two labels are used to denote the patient's state: 'no abnormalities' for normal conditions and 'epileptic seizure' for seizure activity.\n"
    
    for ecg, label in zip(samples, labels):
        if np.isnan(ecg).any():
            continue

        ecg = normalize(ecg.astype(np.float32))
        ecg = ecg.transpose()

        if int(label) == 0:
            text = 'no abnormalities'
        elif int(label) == 1:
            text = 'epileptic seizure'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (178, 1):
            normalized_samples_train.append((text, ecg, label_vector))
    
    for ecg, label in zip(samples_test, labels_test):
        if np.isnan(ecg).any():
            continue
            
        ecg = normalize(ecg.astype(np.float32))
        ecg = ecg.transpose()

        if int(label) == 0:
            text = 'no abnormalities'
        elif int(label) == 1:
            text = 'epileptic seizure'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (178, 1):
            normalized_samples_test.append((text, ecg, label_vector))

    # np.random.shuffle(normalized_samples)
    # split = int(0.8 * len(normalized_samples))
    # samples_train = normalized_samples[:split]
    # samples_test = normalized_samples[split:]

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def process_UWG(data_folder='./data/UWG'):
    normalized_samples_train = []
    normalized_samples_test = []

    samples_train = np.load('./datas/UWG/train_d.npy')
    labels_train = np.load('./datas/UWG/train_l.npy')
    samples_test = np.load('./datas/UWG/test_d.npy')
    labels_test = np.load('./datas/UWG/test_l.npy')
    
    print(labels_test)

    # 遍历数组并规范化
    prefix = "Please choose one of the eight labels to analysis people's gesture based on the provided information.\nThe man is currently showing the gesture of "
    midfix = "A set of eight simple gestures (musical corner, kicking, directing, reversing, ascending and descending arrow, clockwise and countercolockwise loop) generated from accelerometers.\n"
    
    for ecg, label in zip(samples_train, labels_train):
        if np.isnan(ecg).any():
            continue

        ecg = normalize(ecg.astype(np.float32))

        if int(label) == 0:
            text = 'kicking arrow'
        elif int(label) == 1:
            text = 'musical corner'
        elif int(label) == 2:
            text = 'directing arrow'    
        elif int(label) == 3:
            text = 'reversing arrow'
        elif int(label) == 4:
            text = 'ascending arrow'
        elif int(label) == 5:
            text = 'descending arrow'
        elif int(label) == 6:
            text = 'clockwise loop'
        elif int(label) == 7:
            text = 'countercolockwise loop'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (315, 3):
            normalized_samples_train.append((text, ecg, label_vector))
    
    for ecg, label in zip(samples_test, labels_test):
        if np.isnan(ecg).any():
            continue
            
        ecg = normalize(ecg.astype(np.float32))

        if int(label) == 0:
            text = 'kicking arrow'
        elif int(label) == 1:
            text = 'musical corner'
        elif int(label) == 2:
            text = 'directing arrow'    
        elif int(label) == 3:
            text = 'reversing arrow'
        elif int(label) == 4:
            text = 'ascending arrow'
        elif int(label) == 5:
            text = 'descending arrow'
        elif int(label) == 6:
            text = 'clockwise loop'
        elif int(label) == 7:
            text = 'countercolockwise loop'
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (315, 3):
            normalized_samples_test.append((text, ecg, label_vector))

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def process_PS(data_folder='./data/PS'):
    normalized_samples_train = []
    normalized_samples_test = []

    samples_train = np.load('./datas/PS/train_d.npy')
    labels_train = np.load('./datas/PS/train_l.npy')
    samples_test = np.load('./datas/PS/test_d.npy')
    labels_test = np.load('./datas/PS/test_l.npy')

    # 遍历数组并规范化
    prefix = "Please choose one of the thirty-nine labels to analyze the spectrogram characteristics of the phoneme based on the provided information.\nthe man is currently articulating the phoneme "
    midfix = "This data set is a multivaritate representation of a subset of the data used in the paper Dual-domain Hierarchical Classification of Phonetic Time Series.\n"
    
    for ecg, label in zip(samples_train, labels_train):
        if np.isnan(ecg).any():
            continue

        ecg = normalize(ecg.astype(np.float32))

        if int(label) in range(39):
            labels_text = [
                'AH', 'N', 'T', 'L', 'S', 'R', 'IH', 'K', 'IY',
                'D', 'M', 'ER', 'EH', 'P', 'AE', 'B', 'AA', 'EY', 'F', 
                'AY', 'OW', 'SH', 'V', 'G', 'AO', 'Z', 'UW', 
                'NG', 'W', 'JH', 'HH', 'Y', 'CH', 'TH', 'AW', 
                'UH', 'OY', 'DH', 'ZH'
            ]

            text = labels_text[int(label)]
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (217, 11):
            normalized_samples_train.append((text, ecg, label_vector))
    
    for ecg, label in zip(samples_test, labels_test):
        if np.isnan(ecg).any():
            continue
            
        ecg = normalize(ecg.astype(np.float32))

        if int(label) in range(39):
            labels_text = [
                'AH', 'N', 'T', 'L', 'S', 'R', 'IH', 'K', 'IY',
                'D', 'M', 'ER', 'EH', 'P', 'AE', 'B', 'AA', 'EY', 'F', 
                'AY', 'OW', 'SH', 'V', 'G', 'AO', 'Z', 'UW', 
                'NG', 'W', 'JH', 'HH', 'Y', 'CH', 'TH', 'AW', 
                'UH', 'OY', 'DH', 'ZH'
            ]

            text = labels_text[int(label)]
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (217, 11):
            normalized_samples_test.append((text, ecg, label_vector))

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def parse_data_line(line):
    # 将数据行分割为数值部分和标签部分
    data_str, label_str = line.split(':')
    
    # 解析数据部分
    data = list(map(float, data_str.split(',')))
    
    # 解析标签部分
    label = int(label_str)

    return data, label

def read_data_and_labels_from_file(file_path):
    data_points = []
    labels = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # 跳过以 '@' 开头的行
                if line.startswith('@'):
                    continue

                if line:
                    # 解析数据行
                    data, label = parse_data_line(line)

                    # 追加到数据和标签列表
                    data_points.append(data)
                    labels.append(label)

    except FileNotFoundError:
        print(f"错误：未找到文件 '{file_path}'.")

    return data_points, labels

def process_device(data_folder='./data/device'):
    normalized_samples_train = []
    normalized_samples_test = []

    data_val_path = './data/device/val.ts'
    data_train_path = './data/device/FaultDetectionA_TRAIN.ts'
    data_test_path = './data/device/FaultDetectionA_TEST.ts'

    samples_train, labels_train = read_data_and_labels_from_file(data_train_path)
    samples_test, labels_test = read_data_and_labels_from_file(data_test_path)
    samples_val, labels_val = read_data_and_labels_from_file(data_val_path)

    samples_train = np.concatenate([samples_train, samples_val], axis=0)
    labels_train = np.concatenate([labels_train, labels_val], axis=0)

    # 遍历数组并规范化
    prefix = "Selecting one from the above three status, please conduct an analysis of the machine's damage condition in accordance with the provided information.\nThe machine is probably participating in the subsequent damage conditions: "
    midfix = 'These damage conditions include not damaged, inner damaged, and outer damaged.\n'
    for ecg, label in zip(samples_train, labels_train):
        # 检查是否为 NaN 值
        if np.isnan(ecg).any():
            continue  # 丢弃包含 NaN 值的 ECG 数据
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(np.array(ecg).astype(np.float32)).reshape(-1, 1)  # 使用您的规范化函数

        if int(label) == 0:
            text = 'not damaged'
        elif int(label) == 1:
            text = 'inner damaged'
        elif int(label) == 2:
            text = 'outer damaged'    
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (5120, 1):
            normalized_samples_train.append((text, ecg, label_vector))

    for ecg, label in zip(samples_test, labels_test):
        # 检查是否为 NaN 值
        if np.isnan(ecg).any():
            continue  # 丢弃包含 NaN 值的 ECG 数据
            
        # 执行规范化操作，并将结果添加到列表中
        ecg = normalize(np.array(ecg).astype(np.float32)).reshape(-1, 1)  # 使用您的规范化函数

        if int(label) == 0:
            text = 'not damaged'
        elif int(label) == 1:
            text = 'inner damaged'
        elif int(label) == 2:
            text = 'outer damaged'    
        else:
            text = ''

        if text == '':
            continue
        text = midfix + prefix + text  
        label_vector = label
        if ecg.shape == (5120, 1):
            normalized_samples_train.append((text, ecg, label_vector))

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(normalized_samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(normalized_samples_test, file)

    return normalized_samples_train, normalized_samples_test

def modify_text(original_text):
    prefix = "Selecting a previous label based on the provided information.\nThe person's electrocardiogram pattern is indicative of "
    midfix = 'This task focuses on the detection of abnormalities, and it involves classifying the ECGs into two distinct categories: normal and abnormal.\n'
    
    lines = original_text.split('\n')
    side_info = '\n'.join(lines[:2]) + '\n'
    
    if "include(s)" in original_text:
        index = original_text.find("include(s) ")
        original_text = original_text[index + len("include(s) "):] 
    
    if original_text == 'sinus rhythm':
        text = 'normal ecg'
        label = 1
    else:
        text = 'abnormal ecg'
        label = 0
        
    modified_text = side_info + midfix + prefix + text
    return modified_text, label

def process_ecg_bi(Path='./ecg_new'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for i, sample in enumerate(samples_train):
        text, ecg, _ = sample
        modified_text, label = modify_text(text)
        samples_train[i] = (modified_text, ecg, label)

    # 对测试数据集执行相同的操作
    for i, sample in enumerate(samples_test):
        text, ecg, _ = sample
        modified_text, label = modify_text(text)
        samples_test[i] = (modified_text, ecg, label)

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def modify_side_text(original_text):  
    lines = original_text.split('\n')
    modified_text = '\n'.join(lines[2:])
    return modified_text

def process_side_info(Path='./ecg_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for i, sample in enumerate(samples_train):
        text, ecg, label = sample
        modified_text = modify_side_text(text)
        samples_train[i] = (modified_text, ecg, label)

    # 对测试数据集执行相同的操作
    for i, sample in enumerate(samples_test):
        text, ecg, label = sample
        modified_text = modify_side_text(text)
        samples_test[i] = (modified_text, ecg, label)

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def remove_line(original_text, line_number):
    lines = original_text.split('\n')
    if 0 <= line_number < len(lines):
        del lines[line_number]
    modified_text = '\n'.join(lines)
    return modified_text

def process_label_info(Path='./whale_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for i, sample in enumerate(samples_train):
        text, ecg, label = sample
        modified_text = remove_line(text, 0)
        samples_train[i] = (modified_text, ecg, label)

    # 对测试数据集执行相同的操作
    for i, sample in enumerate(samples_test):
        text, ecg, label = sample
        modified_text = remove_line(text, 0)
        samples_test[i] = (modified_text, ecg, label)

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_label_info(Path='./whale_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for i, sample in enumerate(samples_train):
        text, ecg, label = sample
        modified_text = remove_line(text, 0)
        samples_train[i] = (modified_text, ecg, label)

    # 对测试数据集执行相同的操作
    for i, sample in enumerate(samples_test):
        text, ecg, label = sample
        modified_text = remove_line(text, 0)
        samples_test[i] = (modified_text, ecg, label)

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def extract_from_text(text, keyword):
    index = text.find(keyword)
    if index != -1:
        return text[index + len(keyword):] 
    return ""

def extract_all_information(text):
    # 合并搜索，一次性提取所有信息
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

def read_and_modify_last_line(input_string, modification_function):
    # 将输入字符串拆分成行
    lines = input_string.split('\n')
    
    # 如果字符串为空或只有一行，直接替换为修改后的行
    if len(lines) <= 1:
        return modification_function(lines[0])
    
    # 读取最后一行并进行修改
    last_line = lines[-1]
    diagnosis, stage, har, dev, whale = extract_all_information(last_line)
    
    if diagnosis:
        lines[-1] = diagnosis
    elif stage:
        lines[-1] = stage
    elif har:
        lines[-1] = har
    elif dev:
        lines[-1] = dev        
    elif whale:
        lines[-1] = whale        
    
    # 重新组合所有行并返回
    modified_string = '\n'.join(lines)
    return modified_string

def process_word_info(Path='./ecg_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
	
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    for i, sample in enumerate(samples_train):
        text, ecg, label = sample
        modified_text = read_and_modify_last_line(text, 0)
        samples_train[i] = (modified_text, ecg, label)

    # 对测试数据集执行相同的操作
    for i, sample in enumerate(samples_test):
        text, ecg, label = sample
        modified_text = read_and_modify_last_line(text, 0)
        samples_test[i] = (modified_text, ecg, label)

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

import csv

def process_zero_shot_info(Path='./ecg_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
    
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    include = 'include(s) '  # Replace with the actual substring
    label_list = ['atrial flutter', 'bradycardia', 'complete right bundle branch block', 'premature ventricular contractions', 'right axis deviation']

    # Extract and process labels from training data
    extracted_parts_train = []
    for i, sample in enumerate(samples_train):
        label = sample[0]
        index = label.find(include)
        extracted_part = label[index + len(include):]
        extracted_parts_train.append((extracted_part, i))
    
    # Filter samples to be transferred based on label_list
    transfer_indices = [i for label, i in extracted_parts_train if label in label_list]
    transfer_samples = [samples_train[i] for i in transfer_indices]
    samples_train = [sample for i, sample in enumerate(samples_train) if i not in transfer_indices]
    
    # Add the filtered samples to the testing set
    samples_test.extend(transfer_samples)
    
    extracted_parts_train = []
    for i, sample in enumerate(samples_test):
        label = sample[0]
        index = label.find(include)
        extracted_part = label[index + len(include):]
        extracted_parts_train.append((extracted_part, i))
    
    # Filter samples to be transferred based on label_list
    transfer_indices = [i for label, i in extracted_parts_train if label in label_list]
    transfer_samples = [samples_test[i] for i in transfer_indices]
    print(len(transfer_samples))

    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)
    with open('zero_shot.pkl', 'wb') as file:
        pickle.dump(transfer_samples, file)

    return samples_train, samples_test

def process_zero_shot(Path='./ecg_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
    
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    # # 统计训练数据集和测试数据集中的标签数量
    # train_labels = [sample[2] for sample in samples_train]
    # test_labels = [sample[2] for sample in samples_test]
    
    # train_label_counts = Counter(train_labels)
    # test_label_counts = Counter(test_labels)

    # print("训练数据集中的标签数量统计：")
    # for label, count in train_label_counts.items():
    #     print(f"标签 {label}: {count} 个样本")

    # print("\n测试数据集中的标签数量统计：")
    # for label, count in test_label_counts.items():
    #     print(f"标签 {label}: {count} 个样本")

    # 删除训练集中所有标签为1的样本
    samples_train = [sample for sample in samples_train if sample[2] != 2.0]

    # 可以选择保存修改后的训练数据集
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

def process_info(input_string, modification_function):
    # 将输入字符串拆分成行
    lines = input_string.split('\n')
    
    # 如果字符串为空或只有一行，直接替换为修改后的行
    if len(lines) <= 1:
        return modification_function(lines[0])
    
    # 读取最后一行并进行修改
    last_line = lines[-1]
    diagnosis, stage, har, dev, whale = extract_all_information(last_line)
    
    if diagnosis:
        lines[-1] = diagnosis
    elif stage:
        lines[-1] = stage
    elif har:
        lines[-1] = har
    elif dev:
        lines[-1] = dev        
    elif whale:
        lines[-1] = whale        
    
    # 重新组合所有行并返回
    modified_string = '\n'.join(lines)
    return modified_string

def process_gender(Path='./ecg_no_big'):
    train_path = os.path.join(Path, 'samples_train.pkl')
    test_path = os.path.join(Path, 'samples_test.pkl')
    
    samples_train = []
    samples_test = []
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        with open(train_path, 'rb') as file:
            samples_train = pickle.load(file)
        with open(test_path, 'rb') as file:
            samples_test = pickle.load(file)

    # # 统计训练数据集和测试数据集中的标签数量
    # train_labels = [sample[2] for sample in samples_train]
    # test_labels = [sample[2] for sample in samples_test]
    
    # train_label_counts = Counter(train_labels)
    # test_label_counts = Counter(test_labels)

    # print("训练数据集中的标签数量统计：")
    # for label, count in train_label_counts.items():
    #     print(f"标签 {label}: {count} 个样本")

    # print("\n测试数据集中的标签数量统计：")
    # for label, count in test_label_counts.items():
    #     print(f"标签 {label}: {count} 个样本")

    # 删除训练集中所有标签为1的样本
    samples_train = [sample for sample in samples_train if sample[2] != 2.0]

    # 可以选择保存修改后的训练数据集
    with open('samples_train.pkl', 'wb') as file:
        pickle.dump(samples_train, file)
    with open('samples_test.pkl', 'wb') as file:
        pickle.dump(samples_test, file)

    return samples_train, samples_test

if __name__ == "__main__":
    setup_seed()

    samples1, labels1 = process_esr()
    text1, _, _ = samples1[0]

    # samples2, labels2 = process_cpsc_mul()
    # text2, _, _ = samples2[0]

    print(text1)
    print(len(samples1), len(labels1))
    # print(text2)
    # print(len(samples1 + samples2), len(labels1 + labels2))
    # process_ptbxl(few=True)
    # process_12_lead()

    exit(0)