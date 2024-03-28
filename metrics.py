import numpy as np
import pandas as pd
from challeng_score import evaluate_model
from sklearn.metrics import f1_score, hamming_loss

def get_dict(Path):
    mapping_file = Path
    mapping_data = pd.read_csv(mapping_file)

    annotation_to_condition = {}
    for index, row in mapping_data.iterrows():
        annotation_to_condition[row['Full Name']] = index
    
    return annotation_to_condition

def encode_labels(label_dict, label_str, delimiter=','):
    labels = label_str.split(delimiter)
    encoded = [0] * len(label_dict)
    for label in labels:
        label = label.strip()
        if label not in label_dict:
            continue
        encoded[label_dict[label]] = 1

    return encoded

def metric_ecg(preds, labels, logger, delimiter=','):
    diction = get_dict(Path='essy.csv')

    print(preds[0])
    print(labels[0])

    encoded_preds = np.array([encode_labels(diction, p, delimiter) for p in preds])
    encoded_labels = np.array([encode_labels(diction, l, delimiter) for l in labels])
    
    zero_preds = []
    zero_labels = []
    count = 0
    for i, encoded_pred in enumerate(encoded_preds):
        if np.all(encoded_pred == 0):
            zero_preds.append(preds[i])
            zero_labels.append(labels[i])
            count += 1
    
    print(count / len(preds))

    print(encoded_preds[0])
    print(encoded_labels[0])

    hit1 = np.mean(np.all(encoded_preds == encoded_labels, axis=1))
    total_f1 = f1_score(encoded_labels, encoded_preds, average='samples', zero_division=0)
    hloss = hamming_loss(encoded_labels, encoded_preds)
    _, score = evaluate_model(encoded_labels, encoded_preds)
    
    logger.info(
        "Evaluation result:\naccuracy: {}\nTotal F1: {}\nHmloss: {}\nScore: {}\n".format(
            hit1,
            total_f1,
            hloss,
            score
        )
    )

    print(
        "accuracy: {}\nTotal F1: {}\nHmloss: {}\nScore: {}\n".format(
            hit1,
            total_f1,
            hloss,
            score
        )
    )
    return hit1, zero_preds, zero_labels

def metric_eeg(preds_eeg, labels_eeg, logger):
    sleep_stages = {
        'waking up': 0,
        'rapid eye movement sleep': 1,
        'sleep stage one': 2,
        'sleep stage two': 3,
        'sleep stage three': 4,
        'sleep stage four': 5,
        'period of movement': 6,
        'unidentified stage': 7
    }

    print(preds_eeg[0])
    print(labels_eeg[0])

    preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds_eeg])
    labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels_eeg])
    
    zero_preds = []
    zero_labels = []
    count = 0
    for i, encoded_pred in enumerate(preds_mapped):
        if encoded_pred == -1:
            zero_preds.append(preds_eeg[i])
            zero_labels.append(labels_eeg[i])
            count += 1
    
    print(count / len(preds_eeg))

    print(preds_mapped[0])
    print(labels_mapped[0])

    hit2 = np.mean(preds_mapped == labels_mapped)
    sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
    logger.info(
        "Sleep Evaluation result:\naccuracy: {}\nTotal F1 sleep: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    print(
        "accuracy: {}\nTotal F1 sleep: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    return hit2, zero_preds, zero_labels

def metric_har(preds, labels, logger):
    sleep_stages = {
        'walking': 0,
        'ascending stairs': 1,
        'descending stairs': 2,
        'sitting': 3,
        'standing': 4,
        'lying down': 5
    }

    print(preds[0])
    print(labels[0])

    preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
    labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
    zero_preds = []
    zero_labels = []
    count = 0
    for i, encoded_pred in enumerate(preds_mapped):
        if encoded_pred == -1:
            zero_preds.append(preds[i])
            zero_labels.append(labels[i])
            count += 1
    
    print(count / len(preds))

    print(preds_mapped[0])
    print(labels_mapped[0])

    hit2 = np.mean(preds_mapped == labels_mapped)
    sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
    logger.info(
        "HAR Evaluation result:\naccuracy: {}\nTotal F1 HAR: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    print(
        "accuracy: {}\nTotal F1 HAR: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    return hit2, zero_preds, zero_labels

def metric_fd(preds, labels, logger):
    sleep_stages = {
        'not damaged': 0,
        'inner damaged': 1,
        'outer damaged': 2,
    }

    print(preds[0])
    print(labels[0])

    preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
    labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
    zero_preds = []
    zero_labels = []
    count = 0
    for i, encoded_pred in enumerate(preds_mapped):
        if encoded_pred == -1:
            zero_preds.append(preds[i])
            zero_labels.append(labels[i])
            count += 1
    
    print(count / len(preds))
            
    print(preds_mapped[0])
    print(labels_mapped[0])

    hit2 = np.mean(preds_mapped == labels_mapped)
    sleep_f1 = f1_score(labels_mapped, preds_mapped, average='macro', zero_division=0)
    
    logger.info(
        "FD Evaluation result:\naccuracy: {}\nTotal F1 FD: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    print(
        "accuracy: {}\nTotal F1 FD: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    return hit2, zero_preds, zero_labels

def metric_rwc(preds, labels, logger):
    sleep_stages = {
        'the right whale': 0,
        'unknown creature': 1,
    }

    print(preds[0])
    print(labels[0])

    preds_mapped = np.array([sleep_stages.get(stage, -1) for stage in preds])
    labels_mapped = np.array([sleep_stages.get(stage, -1) for stage in labels])
    
    zero_preds = []
    zero_labels = []
    count = 0
    for i, encoded_pred in enumerate(preds_mapped):
        if encoded_pred == -1:
            zero_preds.append(preds[i])
            zero_labels.append(labels[i])
            count += 1
    
    print(count / len(preds))
            
    print(preds_mapped[0])
    print(labels_mapped[0])

    valid_indices = preds_mapped != -1
    valid_preds = preds_mapped[valid_indices]
    valid_labels = labels_mapped[valid_indices]

    hit2 = np.mean(preds_mapped == labels_mapped)
    sleep_f1 = f1_score(valid_labels, valid_preds, average='macro', zero_division=0)
    
    logger.info(
        "RWC Evaluation result:\naccuracy: {}\nTotal F1 RWC: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    print(
        "accuracy: {}\nTotal F1 RWC: {}\n".format(
            hit2,
            sleep_f1
        )
    )

    return hit2, zero_preds, zero_labels