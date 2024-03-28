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

local_model_path = "./gpt2-model"
vqvae_path1 = "./ecg_tokenizer/test_ecg_64_128_40"
vqvae_path2 = "./ecg_tokenizer/test_eeg_64_256_25"
vqvae_path3 = "./ecg_tokenizer/test_fd_64_512_40"
vqvae_path4 = "./ecg_tokenizer/test_har_64_256_1"
vqvae_path5 = "./ecg_tokenizer/test_rwc_64_384_32"

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
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    label_ids = [x["label_ids"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "label_ids": torch.stack(label_ids),
    }

def collate_fn_test(batch):
    input_ids = [x["input_ids"] for x in batch]
    attention_mask = [x["attn_masks"] for x in batch]
    labels = [x["label"] for x in batch]
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": labels,
    }

def test(model, TestDataLoader, args, logger, out=False):
    model.eval()

    with torch.no_grad():
        pred_ids, pred_eeg, pred_har, pred_fd, pred_rwc = [], [], [], [], []
        labels, labels_eeg, labels_har, labels_fd, labels_rwc = [], [], [], [], []

        all_extracted_info = []
        all_sig_labels = []
        if out:
            print_labels = []
            print_preds = []
        for data in tqdm(TestDataLoader, desc="Eval", ncols=120):
            input_ids = data["input_ids"].to(args.device)
            bt_labels = data["labels"]
            
            outputs = model.generate(
                input_ids=input_ids,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=args.num_beams,
                num_return_sequences=args.num_return_sequences,
                do_sample=False,
                max_new_tokens=args.per_max_token,
            )
            
            mask = outputs >= tokenizer.text_vocab_size
            outputs[mask] = tokenizer.pad_token_id
            outputs = outputs[:, args.encoder_max_length:]
            decoded_texts = [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
            all_extracted_info.extend([extract_all_information(dt) for dt in decoded_texts])
            all_sig_labels.extend([extract_all_information(label) for label in bt_labels])
            if out:
                print_labels.extend(bt_labels)
                print_preds.extend(decoded_texts)
        
        for decoded_info, sig_label_info in zip(all_extracted_info, all_sig_labels):
            diagnosis_text, stage_text, har_text, fd_text, rwc_text = decoded_info
            diagnosis_label, stage_label, har_label, fd_label, rwc_label = sig_label_info

            if diagnosis_label:
                pred_ids.append(diagnosis_text)
                labels.append(diagnosis_label)

            elif stage_label:
                pred_eeg.append(stage_text)
                labels_eeg.append(stage_label)

            elif har_label:
                pred_har.append(har_text)
                labels_har.append(har_label)
            
            elif fd_label:
                pred_fd.append(fd_text)
                labels_fd.append(fd_label)

            elif rwc_label:
                pred_rwc.append(rwc_text)
                labels_rwc.append(rwc_label)

        res1, res2, res3, res4, res5 = 0, 0, 0, 0, 0
        if args.dataset == 'mix':
            res1, _, _ = metric_ecg(pred_ids, labels, logger)
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)
            res3, _, _ = metric_har(pred_har, labels_har, logger)
            res4, _, _ = metric_fd(pred_fd, labels_fd, logger)
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)
        elif args.dataset == 'geo':
            res1, _, _ = metric_ecg(pred_ids, labels, logger)
        elif args.dataset == 'eeg':
            res2, _, _ = metric_eeg(pred_eeg, labels_eeg, logger)
        elif args.dataset == 'fd':
            res3, _, _ = metric_fd(pred_fd, labels_fd, logger)
        elif args.dataset == 'rwc':
            res5, _, _ = metric_rwc(pred_rwc, labels_rwc, logger)
        else:
            res4, _, _ = metric_har(pred_har, labels_har, logger)

    if out:
        return print_preds, print_labels
    else:
        return res1 + res2 + res3 + res4 + res5

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
    
    sub_path = "no_frozen"
    
    return model, sub_path

def train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path):
    best = 0.0
        
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
        
        res = test(model, TestDataLoader, args, logger, out=False)
        print(res)
        if res > best:
            MODEL_STORED_PATH = run_path + "/best_model"
            best = res
            model.save_pretrained(MODEL_STORED_PATH)

if __name__ == "__main__":
    args = get_hyperparams()
    seed_everything(args.seed)

    if args.dataset == 'mix' or args.dataset == 'geo':
        file_path = 'ecg_no_big'
        train_path = os.path.join(file_path, 'samples_train.pkl')
        test_path = os.path.join(file_path, 'samples_test.pkl')
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, 'rb') as file:
                samples_train = pickle.load(file)
            with open(test_path, 'rb') as file:
                samples_test = pickle.load(file)
        text1, ecg, _ = samples_train[0]
        print(len(samples_train) + len(samples_test), len(samples_train), len(samples_test))
        print(text1)

    if args.dataset == 'mix' or args.dataset == 'eeg':
        file_path = 'eeg_no_big'
        train_path = os.path.join(file_path, 'samples_train.pkl')
        test_path = os.path.join(file_path, 'samples_test.pkl')
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, 'rb') as file:
                samples_train_eeg = pickle.load(file)
            with open(test_path, 'rb') as file:
                samples_test_eeg = pickle.load(file)
        text2, eeg, _ = samples_train_eeg[0]
        print(len(samples_train_eeg) + len(samples_test_eeg), len(samples_train_eeg), len(samples_test_eeg))
        print(text2)

    if args.dataset == 'mix' or args.dataset == 'fd':
        file_path = 'device_no_big'
        train_path = os.path.join(file_path, 'samples_train.pkl')
        test_path = os.path.join(file_path, 'samples_test.pkl')
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, 'rb') as file:
                samples_train_fd = pickle.load(file)
            with open(test_path, 'rb') as file:
                samples_test_fd = pickle.load(file)
        text3, fd, _ = samples_train_fd[0]
        print(len(samples_train_fd) + len(samples_test_fd), len(samples_train_fd), len(samples_test_fd))
        print(text3)

    if args.dataset == 'mix' or args.dataset == 'har':
        file_path = 'har_no_big'
        train_path = os.path.join(file_path, 'samples_train.pkl')
        test_path = os.path.join(file_path, 'samples_test.pkl')
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, 'rb') as file:
                samples_train_har = pickle.load(file)
            with open(test_path, 'rb') as file:
                samples_test_har = pickle.load(file)
        text4, har, _ = samples_train_har[0]
        print(len(samples_train_har) + len(samples_test_har), len(samples_train_har), len(samples_test_har))
        print(text4)
        
    if args.dataset == 'mix' or args.dataset == 'rwc':
        file_path = 'rwc_no_big'
        train_path = os.path.join(file_path, 'samples_train.pkl')
        test_path = os.path.join(file_path, 'samples_test.pkl')
        if os.path.isfile(train_path) and os.path.isfile(test_path):
            with open(train_path, 'rb') as file:
                samples_train_rwc = pickle.load(file)
            with open(test_path, 'rb') as file:
                samples_test_rwc = pickle.load(file)
        text7, rwc, _ = samples_train_rwc[0]
        print(len(samples_train_rwc) + len(samples_test_rwc), len(samples_train_rwc), len(samples_test_rwc))
        print(text7)
        
    print('preprocess done')

    if args.dataset == 'mix':
        samples_train_combined = samples_train + samples_train_eeg + samples_train_har + samples_train_fd + samples_train_rwc
        samples_test_combined = samples_test + samples_test_eeg + samples_test_har + samples_test_fd + samples_test_rwc
        np.random.shuffle(samples_train_combined)
        np.random.shuffle(samples_test_combined)
        PREFIX_TEXT = "You will be receiving signals from five domains: electrocardiogram, electroencephalogram, industrial equipment, sound and physical activities.\n"
    elif args.dataset == 'geo':
        samples_train_combined = samples_train
        samples_test_combined = samples_test
        PREFIX_TEXT = "You will be receiving electrocardiogram(ECG) related signals.\n"
    elif args.dataset == 'eeg':
        samples_train_combined = samples_train_eeg
        samples_test_combined = samples_test_eeg
        PREFIX_TEXT = "You will be receiving electroencephalogram(EEG) related signals.\n"
    elif args.dataset == 'fd':
        samples_train_combined = samples_train_fd
        samples_test_combined = samples_test_fd
        PREFIX_TEXT = "You will be receiving industrial equipment related signals.\n"
    elif args.dataset == 'rwc':
        samples_train_combined = samples_train_rwc
        samples_test_combined = samples_test_rwc
        PREFIX_TEXT = "You will be receiving sound related signals.\n"
    else:
        samples_train_combined = samples_train_har
        samples_test_combined = samples_test_har
        PREFIX_TEXT = "You will be receiving human physical activities related signals.\n"

    TStokenizer1 = load_TStokenizer(vqvae_path1, ecg.shape, 'cpu')
    TStokenizer2 = load_TStokenizer(vqvae_path2, eeg.shape, 'cpu')
    TStokenizer3 = load_TStokenizer(vqvae_path3, fd.shape, 'cpu')
    TStokenizer4 = load_TStokenizer(vqvae_path4, har.shape, 'cpu')
    TStokenizer5 = load_TStokenizer(vqvae_path5, rwc.shape, 'cpu')
    TStokenizers = [TStokenizer1, TStokenizer2, TStokenizer3, TStokenizer4, TStokenizer5]
    tokenizer = MultiTokenizer(TStokenizers)

    TrainDataset = MultiDataset(
        samples_train_combined,
        tokenizer,
        mode="train",
        encoder_max_length=args.encoder_max_length,
        multi=args.dataset,
        prefix_text=PREFIX_TEXT,
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
        multi=args.dataset,
        prefix_text=PREFIX_TEXT,
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
        
        if args.adapt:
            best_model_path = os.path.join(run_path, 'best_model')
            model_state_dict = torch.load(os.path.join(args.load_model_path, 'pytorch_model.bin'), map_location=args.device)
            model.load_state_dict(model_state_dict, strict=False)

        for param in model.parameters():
            param.requires_grad = True
            
        param_dict = [{"params": model.parameters(), "lr": args.lr}]
        optimizer = torch.optim.Adam(param_dict, weight_decay=1e-5)
        scheduler = transformers.optimization.get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=args.epochs * len(TrainDataLoader) * args.warm_up_ratio, num_training_steps=args.epochs * len(TrainDataLoader)
        )
        scaler = GradScaler()

        logger.info(f"Begin training for run {run}")
        train_model(model, args, TrainDataLoader, TestDataLoader, optimizer, scheduler, scaler, logger, run_path)

        model, _ = initialize_model(args, tokenizer)
        best_model_path = os.path.join(run_path, 'best_model')
        model_state_dict = torch.load(os.path.join(best_model_path, 'pytorch_model.bin'), map_location=args.device)
        model.load_state_dict(model_state_dict)

        logger.info(f"Test best model for run {run}")
        print_preds, print_labels = test(model, TestDataLoader, args, logger, out=True)

        save_path = os.path.join(run_path, 'output.txt')
        with open(save_path, 'w', encoding='utf-8') as file:
            if args.dataset == 'mix' or args.dataset == 'geo':
                file.write("Input Sequence: \n{}\n".format(PREFIX_TEXT + text1))
                file.write('\n')
            if args.dataset == 'mix' or args.dataset == 'eeg':
                file.write("Input Sequence: \n{}\n".format(PREFIX_TEXT + text2))
                file.write('\n')
            if args.dataset == 'mix' or args.dataset == 'fd':
                file.write("Input Sequence: \n{}\n".format(PREFIX_TEXT + text3))
                file.write('\n')
            if args.dataset == 'mix' or args.dataset == 'har':
                file.write("Input Sequence: \n{}\n".format(PREFIX_TEXT + text4))
                file.write('\n')
            if args.dataset == 'mix' or args.dataset == 'rwc':
                file.write("Input Sequence: \n{}\n".format(PREFIX_TEXT + text7))
                file.write('\n')

            for i in range(500):
                j = i * args.num_return_sequences
                for k in range(args.num_return_sequences):
                    file.write("Generated Text: {}\n".format(print_preds[j + k]))
                file.write("Actual Label: {}\n".format(print_labels[i]))
                file.write('\n')

        logger.handlers.clear()