import os
import random
import logging
from logging.handlers import RotatingFileHandler
import pickle
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import transformers
from transformers import GPT2Config, GPT2LMHeadModel

from multimodel import InstructTime, MultiTokenizer
from multidataset import MultiDataset
from args import get_hyperparams
from utils import load_TStokenizer


# 固定本地模型路径（与现有脚本保持一致）
local_model_path = "./gpt2"

DATA_ROOT = "./datasets"
VQVAE_ROOT = "./vqvae"

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

DATASET_CONFIG: Dict[str, Dict[str, List[str]]] = {
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


def _find_dataset_split(subdirs: List[str]) -> Tuple[str, str, str]:
    candidates = list(_expand_candidates(subdirs, DATA_ROOT))
    checked = []
    for candidate in candidates:
        train_path = os.path.join(candidate, "samples_train.pkl")
        test_path = os.path.join(candidate, "samples_test.pkl")
        checked.append(candidate)
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            return candidate, train_path, test_path
    raise FileNotFoundError(f"Missing dataset split. Checked directories: {checked}")


def _find_tokenizer_dir(subdirs: List[str]) -> str:
    candidates = list(_expand_candidates(subdirs, VQVAE_ROOT))
    checked = []
    for candidate in candidates:
        checked.append(candidate)
        args_path = os.path.join(candidate, "args.json")
        model_path = os.path.join(candidate, "model.pkl")
        if os.path.isfile(args_path) and os.path.isfile(model_path):
            return candidate
    raise FileNotFoundError(f"Tokenizer path not found. Checked directories: {checked}")


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


def load_dataset_bundle(key: str, log_messages=None):
    if key not in DATASET_CONFIG:
        raise ValueError(f"Unsupported dataset '{key}'. Available: {list(DATASET_CONFIG.keys())}")
    info = DATASET_CONFIG[key]
    data_dir, train_path, test_path = _find_dataset_split(info["data_subdirs"])
    with open(train_path, "rb") as f:
        train_samples = pickle.load(f)
    with open(test_path, "rb") as f:
        test_samples = pickle.load(f)
    example_text, signal = extract_text_signal(train_samples[0])

    tokenizer_path = _find_tokenizer_dir(info["vqvae_subdirs"])
    if log_messages is not None:
        log_messages.extend(
            [
                f"{key.upper()} train file: {train_path}",
                f"{key.upper()} test file: {test_path}",
                f"{key.upper()} tokenizer path: {tokenizer_path}",
                f"{key.upper()} samples: total {len(train_samples) + len(test_samples)} "
                f"(train {len(train_samples)}, test {len(test_samples)})",
            ]
        )

    return {
        "key": key,
        "train": train_samples,
        "test": test_samples,
        "text": example_text,
        "signal_shape": tuple(np.asarray(signal).shape),
        "tokenizer_path": tokenizer_path,
        "prefix": info["prefix"],
    }


def setup_reproducible_runtime():
    """尽量消除随机性，提升可复现性。"""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
    os.environ.setdefault("NCCL_TIMEOUT", "1800")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        from torch.backends.cuda import sdp_kernel
        sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
    except Exception:
        pass
    try:
        torch.use_deterministic_algorithms(True)
        if hasattr(torch, "set_deterministic_debug_mode"):
            torch.set_deterministic_debug_mode("warn")
    except Exception:
        pass


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def collate_fn_pretrain(batch):
    """跨域自回归预训练：右填充并将标签等于输入。"""
    pad_token_id = batch[0]["pad_token_id"]
    input_seqs = [x["input_ids"] for x in batch]
    max_len = max(seq.size(0) for seq in input_seqs)
    bsz = len(batch)

    input_ids = torch.full((bsz, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.full((bsz, max_len), -100, dtype=torch.long)

    for i, seq in enumerate(input_seqs):
        L = seq.size(0)
        input_ids[i, :L] = seq
        attention_mask[i, :L] = 1
        labels[i, :L] = seq

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def setup_logging(run_path: str):
    log_file = os.path.join(run_path, "log.log")
    os.makedirs(run_path, exist_ok=True)
    open(log_file, "w").close()
    logger = logging.getLogger(f"pretrain_log_{run_path}")
    logger.setLevel(logging.INFO)
    file_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024 * 5, backupCount=2)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    return logger


def save_model_checkpoint(model, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(save_dir)
    torch.save(model_to_save.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))


def initialize_model(args, tokenizer, TStokenizers):
    """构建多模态模型并加载 GPT-2 预训练权重，替换输出头。"""
    seed_everything(args.seed)

    config = GPT2Config.from_pretrained(local_model_path)
    model = InstructTime(config, TStokenizers, text_embedding=len(tokenizer.textTokenizer)).to(args.device)

    pretrained_gpt2_model = GPT2LMHeadModel.from_pretrained(local_model_path)
    model.load_state_dict(pretrained_gpt2_model.state_dict(), strict=False)

    model.resize_token_embeddings(len(tokenizer.textTokenizer))
    current_output = model.get_output_embeddings()
    new_output = torch.nn.Linear(config.n_embd, tokenizer.vocabSize_all(), bias=False).to(args.device)
    new_output.weight.data[: len(tokenizer.textTokenizer)] = current_output.weight.data
    model.set_output_embeddings(new_output)
    model.config.vocab_size = tokenizer.vocabSize_all()

    sub_path = "cross_domain_pretrain"
    return model, sub_path


def train_pretrain_model(model, args, TrainDataLoader, optimizer, scheduler, scaler, logger, run_path):
    best_loss = float("inf")
    patience = 10
    wait = 0

    for epoch in range(args.epochs):
        step, train_losses = 0, 0.0
        tqdm_iter = tqdm(TrainDataLoader, desc=f"Pretrain Epoch {epoch+1}", ncols=120)

        model.train()
        for data in tqdm_iter:
            input_ids = data["input_ids"].to(args.device)
            attention_mask = data["attention_mask"].to(args.device)
            labels = data["labels"].to(args.device)

            with autocast():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = outputs.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()

            loss_value = float(loss.detach().cpu())
            train_losses += loss_value
            step += 1
            tqdm_iter.set_postfix({"loss": f"{train_losses/step:.4f}", "lr": f"{current_lr:.2e}"})

        avg_loss = train_losses / max(step, 1)
        logger.info(f"Epoch {epoch+1} | Pretrain Loss: {avg_loss:.4f}; lr={current_lr:.6e}")
        print(f"[Pretrain] Epoch {epoch+1} Loss {avg_loss:.4f} | lr {current_lr:.6e}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            wait = 0
            save_dir = os.path.join(run_path, "best_model")
            save_model_checkpoint(model, save_dir)
            logger.info(f"New best loss {best_loss:.4f}, model saved to {save_dir}")
        else:
            wait += 1
            if wait >= patience:
                logger.info(f"Early stopping at epoch {epoch+1} with no loss improvement for {patience} epochs")
                break

    return best_loss


if __name__ == "__main__":
    setup_reproducible_runtime()
    args = get_hyperparams()

    os.environ["PYTHONHASHSEED"] = str(args.seed)
    seed_everything(args.seed)

    DATA_ROOT = args.data_root
    VQVAE_ROOT = args.vqvae_root

    dataset_key = normalize_dataset_key(args.dataset)
    if dataset_key != "mix":
        raise ValueError("Cross-domain 预训练仅支持 --dataset mix。")

    log_messages: List[str] = []
    bundles = [load_dataset_bundle(key, log_messages) for key in MIX_ORDER]

    samples_train_combined = []
    domain_weights = []
    for bundle in bundles:
        size = len(bundle["train"])
        weight = 1.0 / max(size, 1)
        samples_train_combined.extend(bundle["train"])
        domain_weights.extend([weight] * size)

    TStokenizers = []
    for bundle in bundles:
        tokenizer_path = bundle["tokenizer_path"]
        TStokenizers.append(load_TStokenizer(tokenizer_path, bundle["signal_shape"], "cpu"))
    tokenizer = MultiTokenizer(TStokenizers, dataset_keys=MIX_ORDER)

    TrainDataset = MultiDataset(
        samples_train_combined,
        tokenizer,
        mode="train",
        encoder_max_length=args.encoder_max_length,
        multi=args.dataset,
        prefix_text=MIX_PREFIX,
    )

    dl_seed_gen = torch.Generator()
    dl_seed_gen.manual_seed(int(args.seed))
    _NUM_WORKERS = 16

    sample_weights = torch.DoubleTensor(domain_weights)
    train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    TrainDataLoader = DataLoader(
        TrainDataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False,
        num_workers=_NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        generator=dl_seed_gen,
        collate_fn=collate_fn_pretrain,
    )

    model, sub_path = initialize_model(args, tokenizer, TStokenizers)

    model_subpath = os.path.join(args.model_path, sub_path)
    os.makedirs(model_subpath, exist_ok=True)
    run_path = os.path.join(model_subpath, "pretrain_run_0")
    os.makedirs(run_path, exist_ok=True)
    logger = setup_logging(run_path)
    for msg in log_messages:
        logger.info(msg)

    param_dict = [{"params": model.parameters(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(param_dict, weight_decay=0.01)
    scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.epochs * len(TrainDataLoader) * args.warm_up_ratio,
        num_training_steps=args.epochs * len(TrainDataLoader),
    )
    scaler = GradScaler()

    logger.info("Begin cross-domain autoregressive pretraining")
    train_pretrain_model(
        model,
        args,
        TrainDataLoader,
        optimizer,
        scheduler,
        scaler,
        logger,
        run_path,
    )

    logger.info("Pretraining finished")
    print("Cross-domain pretraining finished.")
