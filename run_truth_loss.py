import os
import torch
import random
import logging
from logging.handlers import RotatingFileHandler
import pickle
import transformers
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import (
    GPT2LMHeadModel,
    GPT2Config,
)

from multimodel import InstructTime, MultiTokenizer
from multidataset import MultiDataset
from args import get_hyperparams
from metrics import metric_ecg, metric_eeg, metric_har, metric_fd, metric_rwc
from utils import extract_all_information, load_TStokenizer

local_model_path = "./gpt2"

DATASET_ALIAS_MAP = {
    "sleep": "eeg",
    "ecg": "geo",
    "dev": "fd",
    "whale": "rwc",
}

MIX_ORDER = ["geo", "eeg", "fd", "har", "rwc"]
MIX_PREFIX = (
    "You will be receiving signals from five domains: electrocardiogram, "
    "electroencephalogram, industrial equipment, sound and physical activities.\n"
)

DATASET_CONFIG = {
    "geo": {
        "data_subdirs": ["GEO", "geo", "ECG", "ecg_no_big"],
        "vqvae_subdirs": ["ECG", "ecg", "GEO", "geo"],
        "prefix": "You will be receiving electrocardiogram(ECG) related signals.\n",
    },
    "eeg": {
        "data_subdirs": ["EEG", "eeg", "eeg_no_big"],
        "vqvae_subdirs": ["EEG", "eeg"],
        "prefix": "You will be receiving electroencephalogram(EEG) related signals.\n",
    },
    "fd": {
        "data_subdirs": ["FD", "fd", "device_no_big"],
        "vqvae_subdirs": ["FD", "fd"],
        "prefix": "You will be receiving industrial equipment related signals.\n",
    },
    "har": {
        "data_subdirs": ["HAR", "har", "har_no_big"],
        "vqvae_subdirs": ["HAR", "har"],
        "prefix": "You will be receiving human physical activities related signals.\n",
    },
    "rwc": {
        "data_subdirs": ["RWC", "rwc", "rwc_no_big"],
        "vqvae_subdirs": ["RWC", "rwc"],
        "prefix": "You will be receiving sound related signals.\n",
    },
}


def normalize_dataset_key(name: str) -> str:
    ds = (name or "").lower()
    return DATASET_ALIAS_MAP.get(ds, ds)


def _expand_candidates(subdirs, base_dir):
    for sub in subdirs:
        if os.path.isabs(sub):
            yield sub
        else:
            yield os.path.join(base_dir, sub)


def _find_dataset_split(subdirs):
    candidates = list(_expand_candidates(subdirs, DATA_ROOT))
    checked = []
    for candidate in candidates:
        train_path = os.path.join(candidate, "samples_train.pkl")
        test_path = os.path.join(candidate, "samples_test.pkl")
        checked.append(candidate)
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            return candidate, train_path, test_path
    raise FileNotFoundError(f"Missing dataset split. Checked: {checked}")


def _find_tokenizer_dir(subdirs):
    candidates = list(_expand_candidates(subdirs, VQVAE_ROOT))
    checked = []
    for candidate in candidates:
        args_path = os.path.join(candidate, "args.json")
        model_path = os.path.join(candidate, "model.pkl")
        checked.append(candidate)
        if os.path.isfile(args_path) and os.path.isfile(model_path):
            return candidate
    raise FileNotFoundError(f"Tokenizer path not found. Checked: {checked}")


def extract_text_signal(sample):
    if isinstance(sample, dict):
        text = sample.get("text", "")
        signal = sample.get("ts")
    elif isinstance(sample, (list, tuple)) and len(sample) >= 2:
        text = sample[0]
        signal = sample[1]
    else:
        raise ValueError("Unsupported sample format; expected dict or tuple/list with text and signal")
    if signal is None:
        raise ValueError("Signal tensor missing in sample; ensure dataset contains (text, ts, label) entries")
    return text, signal


def load_dataset_bundle(key, log_messages=None):
    if key not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{key}'. Available: {list(DATASET_CONFIG.keys())}")
    info = DATASET_CONFIG[key]
    data_dir, train_path, test_path = _find_dataset_split(info["data_subdirs"])
    with open(train_path, "rb") as f:
        train_samples = pickle.load(f)
    with open(test_path, "rb") as f:
        test_samples = pickle.load(f)
    text_example, signal = extract_text_signal(train_samples[0])
    total_msg = (
        f"{key.upper()} samples: total {len(train_samples) + len(test_samples)} "
        f"(train {len(train_samples)}, test {len(test_samples)})"
    )
    if log_messages is not None:
        log_messages.extend(
            [
                total_msg,
                f"{key.upper()} train file: {train_path}",
                f"{key.upper()} test file: {test_path}",
                f"{key.upper()} example text: {text_example}",
            ]
        )
    tokenizer_path = _find_tokenizer_dir(info["vqvae_subdirs"])
    return {
        "key": key,
        "train": train_samples,
        "test": test_samples,
        "text": text_example,
        "signal_shape": tuple(np.asarray(signal).shape),
        "prefix": info["prefix"],
        "tokenizer_path": tokenizer_path,
    }


def collect_task_predictions(decoded_texts, label_texts):
    preds = {k: [] for k in MIX_ORDER}
    labels = {k: [] for k in MIX_ORDER}
    for pred_text, lbl_text in zip(decoded_texts, label_texts):
        diagnosis_text, stage_text, har_text, dev_text, whale_text = extract_all_information(pred_text)
        diagnosis_lbl, stage_lbl, har_lbl, dev_lbl, whale_lbl = extract_all_information(lbl_text)
        if diagnosis_lbl:
            preds["geo"].append(diagnosis_text)
            labels["geo"].append(diagnosis_lbl)
        elif stage_lbl:
            preds["eeg"].append(stage_text)
            labels["eeg"].append(stage_lbl)
        elif har_lbl:
            preds["har"].append(har_text)
            labels["har"].append(har_lbl)
        elif dev_lbl:
            preds["fd"].append(dev_text)
            labels["fd"].append(dev_lbl)
        elif whale_lbl:
            preds["rwc"].append(whale_text)
            labels["rwc"].append(whale_lbl)
    return preds, labels


def evaluate_task_metrics(dataset_key, preds_dict, labels_dict, logger):
    key = dataset_key.lower()
    metric_map = {
        "geo": metric_ecg,
        "eeg": metric_eeg,
        "har": metric_har,
        "fd": metric_fd,
        "rwc": metric_rwc,
    }
    if key == "mix":
        total = 0.0
        for sub_key in MIX_ORDER:
            preds = preds_dict.get(sub_key, [])
            lbls = labels_dict.get(sub_key, [])
            if not preds or not lbls:
                continue
            score, _, _ = metric_map[sub_key](preds, lbls, logger)
            total += score
        return total
    if key not in metric_map:
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Supported: {list(metric_map.keys()) + ['mix']}")
    preds = preds_dict.get(key, [])
    lbls = labels_dict.get(key, [])
    if not preds or not lbls:
        return 0.0
    score, _, _ = metric_map[key](preds, lbls, logger)
    return score

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

def collate_fn_train(batch):
    """Train: dynamic right padding (standard GPT-2 style).
    Content stays on the left, padding on the right; label padding positions are -100 and ignored in loss.
    """
    pad_token_id = batch[0]["pad_token_id"]
    input_seqs = [x["input_ids"] for x in batch]
    label_seqs = [x["label_ids"] for x in batch]
    max_len = max(seq.size(0) for seq in input_seqs)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    label_ids = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, (inp, lab) in enumerate(zip(input_seqs, label_seqs)):
        L = inp.size(0)
        # Right padding: place content on the left (positions 0 to L-1)
        input_ids[i, :L] = inp
        attention_mask[i, :L] = 1
        label_ids[i, :L] = lab

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "label_ids": label_ids,
    }

def collate_fn_test(batch):
    """Inference: dynamic left padding (recommended for decoder-only generate).
    Avoids the \"right-padding was detected\" warning and improves logits alignment.
    """
    pad_token_id = batch[0]["pad_token_id"]
    input_seqs = [x["input_ids"] for x in batch]
    labels = [x["label"] for x in batch]
    max_len = max(seq.size(0) for seq in input_seqs)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    for i, inp in enumerate(input_seqs):
        L = inp.size(0)
        # Left padding: place content on the right (positions max_len-L to max_len-1)
        input_ids[i, max_len - L:] = inp
        attention_mask[i, max_len - L:] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def test(model, TestDataLoader, args, logger, tokenizer, out=False):
    model.eval()

    preds_by_task = {k: [] for k in MIX_ORDER}
    labels_by_task = {k: [] for k in MIX_ORDER}
    print_labels, print_preds = [], []

    with torch.no_grad():
        for data in tqdm(TestDataLoader, desc="Eval", ncols=120):
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            bt_labels = data["labels"]

            # Use a decoding strategy close to the original GPT-2 setup without extra penalties to avoid abnormal truncation
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                max_new_tokens=args.per_max_token,
                use_cache=True,
            )

            mask = outputs >= tokenizer.text_vocab_size
            outputs[mask] = tokenizer.pad_token_id
            # Prompt length equals the full input length (including left padding)
            prompt_len = input_ids.shape[1]
            outputs = outputs[:, prompt_len:]
            decoded_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            batch_preds, batch_labels = collect_task_predictions(decoded_texts, bt_labels)
            for key in MIX_ORDER:
                preds_by_task[key].extend(batch_preds[key])
                labels_by_task[key].extend(batch_labels[key])
            if out:
                print_labels.extend(bt_labels)
                print_preds.extend(decoded_texts)

    if out:
        return print_preds, print_labels
    return evaluate_task_metrics(args.dataset_key, preds_by_task, labels_by_task, logger)

def setup_logging(run_path):
    """
    logger
    """
    log_file = os.path.join(run_path, "log.log")


    open(log_file, 'w').close()
    logger = logging.getLogger('training_log')
    logger.setLevel(logging.INFO)

    file_handler = RotatingFileHandler(log_file, maxBytes=1024*1024*5, backupCount=2)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger


def save_model_checkpoint(model, save_dir):
    """
    Save model weights in Hugging Face format and ensure pytorch_model.bin exists.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)
    # Explicitly save state_dict to guarantee bin file exists
    torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

def initialize_model(args, tokenizer, TStokenizers):
    config = GPT2Config.from_pretrained(local_model_path)
    model = InstructTime(config, TStokenizers, text_embedding=len(tokenizer.textTokenizer)).to(args.device)

    pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained(local_model_path)
    model.load_state_dict(pretrained_gpt2_model.state_dict(), strict=False)

    model.resize_token_embeddings(len(tokenizer.textTokenizer))
    current_output = model.get_output_embeddings()
    new_output = nn.Linear(config.n_embd, tokenizer.vocabSize_all(), bias=False).to(args.device)
    new_output.weight.data[:len(tokenizer.textTokenizer)] = current_output.weight.data
    model.set_output_embeddings(new_output)
    model.config.vocab_size = tokenizer.vocabSize_all()
    
    sub_path = "no_frozen"
    
    return model, sub_path

def train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path, tokenizer):
    best = -float("inf")
        
    for epoch in range(args.epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(TrainDataLoader, desc=f"GPT Epoch {epoch+1}", ncols=120)
        
        model.train()
        for data in tqdm_iter:

            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            label_ids = data["label_ids"].to(args.device)
            
            with autocast():
                outputs = model(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=label_ids
                            )
            
            scaler.scale(outputs.loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            loss_value = outputs.loss.cpu().item()
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": format(train_losses / step, ".4f")})

        final_loss = format(train_losses / step, ".4f")
        logger.info(f"Epoch {epoch+1}\nLoss: {final_loss}")
        
        res = test(model, TestDataLoader, args, logger, tokenizer, out=False)
        print(res)
        if res > best:
            MODEL_STORED_PATH = run_path + "/best_model"
            best = res
            save_model_checkpoint(model, MODEL_STORED_PATH)

if __name__ == "__main__":
    args = get_hyperparams()
    seed_everything(args.seed)
    DATA_ROOT = args.data_root
    VQVAE_ROOT = args.vqvae_root

    dataset_key = normalize_dataset_key(args.dataset)
    args.dataset_key = dataset_key
    if dataset_key != "mix" and dataset_key not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{args.dataset}'. Supported: {list(DATASET_CONFIG.keys()) + ['mix']}")

    log_messages = []
    selected_keys = MIX_ORDER if dataset_key == "mix" else [dataset_key]
    bundle_map = {}
    for key in selected_keys:
        bundle_map[key] = load_dataset_bundle(key, log_messages)

    if dataset_key == "mix":
        samples_train_combined = []
        samples_test_combined = []
        for key in selected_keys:
            samples_train_combined.extend(bundle_map[key]["train"])
            samples_test_combined.extend(bundle_map[key]["test"])
        random.shuffle(samples_train_combined)
        random.shuffle(samples_test_combined)
        PREFIX_TEXT = MIX_PREFIX
    else:
        bundle = bundle_map[selected_keys[0]]
        samples_train_combined = bundle["train"]
        samples_test_combined = bundle["test"]
        PREFIX_TEXT = bundle["prefix"]

    example_inputs = [(key, bundle_map[key]["text"], bundle_map[key]["prefix"]) for key in selected_keys]

    # Only load tokenizers required for the current task to avoid an unnecessarily large vocabulary for single-domain training
    tokenizer_keys = MIX_ORDER if dataset_key == "mix" else selected_keys
    TStokenizers = [
        load_TStokenizer(bundle_map[key]["tokenizer_path"], bundle_map[key]["signal_shape"], "cpu")
        for key in tokenizer_keys
    ]
    tokenizer = MultiTokenizer(TStokenizers, dataset_keys=tokenizer_keys)

    TrainDataset = MultiDataset(
        samples_train_combined,
        tokenizer,
        mode="train",
        encoder_max_length=args.encoder_max_length,
        multi=args.dataset_key,
        prefix_text=PREFIX_TEXT,
        dataset_key=args.dataset_key if dataset_key != "mix" else None,
    )
    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_train,
    )
    TestDataset = MultiDataset(
        samples_test_combined,
        tokenizer,
        mode="test",
        encoder_max_length=args.encoder_max_length,
        multi=args.dataset_key,
        prefix_text=PREFIX_TEXT,
        dataset_key=args.dataset_key if dataset_key != "mix" else None,
    )
    TestDataLoader = DataLoader(
        TestDataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn_test,
    )

    num = 1
    for run in range(num):
        model, sub_path = initialize_model(args, tokenizer, TStokenizers)
        model_subpath = os.path.join(args.model_path, sub_path)
        print(args.model_path, model_subpath)

        os.makedirs(model_subpath, exist_ok=True)
        run_path = os.path.join(model_subpath, f"run_{run}")
        os.makedirs(run_path, exist_ok=True)
        logger = setup_logging(run_path)
        if log_messages:
            for msg in log_messages:
                logger.info(msg)
        
        if args.adapt:
            model_state_dict = torch.load(os.path.join(args.load_model_path, 'pytorch_model.bin'), map_location=args.device)
            model.load_state_dict(model_state_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = True
            
        param_dict = [{"params": model.parameters(), "lr": args.lr}]
        optimizer = torch.optim.AdamW(param_dict, weight_decay=0.01)
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.epochs * len(TrainDataLoader) * args.warm_up_ratio, num_training_steps=args.epochs * len(TrainDataLoader)
        )
        scaler = GradScaler()

        logger.info(f"Begin training for run {run}")
        train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path, tokenizer)

        model, _ = initialize_model(args, tokenizer, TStokenizers)
        best_model_path = os.path.join(run_path, 'best_model')
        state_file = os.path.join(best_model_path, 'pytorch_model.bin')
        if not os.path.exists(state_file):
            save_model_checkpoint(model, best_model_path)
        model_state_dict = torch.load(state_file, map_location=args.device)
        model.load_state_dict(model_state_dict)

        logger.info(f"Test best model for run {run}")
        print_preds, print_labels = test(model, TestDataLoader, args, logger, tokenizer, out=True)

        save_path = os.path.join(run_path, 'output.txt')
        with open(save_path, 'w', encoding='utf-8') as file:
            for name, example_text, prefix in example_inputs:
                file.write(f"Input Sequence ({name.upper()}): \n{prefix + example_text}\n\n")

            limit = min(500, len(print_labels))
            for i in range(limit):
                j = i * args.num_return_sequences
                for k in range(args.num_return_sequences):
                    file.write("Generated Text: {}\n".format(print_preds[j + k]))
                file.write("Actual Label: {}\n\n".format(print_labels[i]))

        logger.handlers.clear()
