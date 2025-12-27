<div align="center">
  <h1><b>InstructTime</b></h1>
  <h3>InstrucTime: Advancing Time Series Classification with Multimodal Language Modeling</h3>
  <p><b>ACM WSDM 2025</b></p>
  
  <a href="https://dl.acm.org/doi/abs/10.1145/3701551.3703499"><img src="https://img.shields.io/badge/Paper-ACM%20DL-blue" alt="Paper"></a>
  <a href="https://huggingface.co/datasets/zhjai/InstructTime/tree/main"><img src="https://img.shields.io/badge/🤗%20Dataset-InstructTime-yellow" alt="Dataset"></a>
  <a href="https://huggingface.co/openai-community/gpt2"><img src="https://img.shields.io/badge/🤗%20Model-GPT--2-orange" alt="Model"></a>
</div>

---

> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 
> 🌟 If you find this resource helpful, please consider starring this repository and citing our research.

## Citation

```bibtex
@inproceedings{cheng2025instructime,
  title={Instructime: Advancing time series classification with multimodal language modeling},
  author={Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong and Chen, Enhong},
  booktitle={Proceedings of the Eighteenth ACM International Conference on Web Search and Data Mining},
  pages={792--800},
  year={2025}
}
```

## Overview

<img width="1279" height="432" alt="InstructTime Architecture" src="https://github.com/user-attachments/assets/d0a34379-b1f3-434f-a7ae-fac69f843207" />

InstructTime is a multimodal language model for time series classification that bridges the gap between time series data and natural language understanding.

## Resources

| Resource | Link |
|----------|------|
| 🤗 Dataset | [zhjai/InstructTime](https://huggingface.co/datasets/zhjai/InstructTime) |
| 🤗 Base Model | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) |
| 📄 Paper | [ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3701551.3703499) |

## Dataset Name Mapping

The following table shows the mapping between dataset names used in the code and their corresponding domains:

| Code Name | Domain | Description |
|-----------|--------|-------------|
| `sleep` | EEG | Electroencephalogram (Sleep Stage) |
| `geo` / `ecg` | ECG | Electrocardiogram |
| `dev` | FD | Fault Detection (Industrial Equipment) |
| `har` | HAR | Human Activity Recognition |
| `whale` | RWC | Real World Computing (Whale Sound) |

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.1+
- CUDA (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/InstructTime.git
cd InstructTime

# Install dependencies
pip install -r requirements.txt

# Download GPT-2 model from HuggingFace (required)
# Option 1: Using huggingface-cli
huggingface-cli download openai-community/gpt2 --local-dir ./gpt2

# Option 2: Using git lfs
git lfs install
git clone https://huggingface.co/openai-community/gpt2 ./gpt2
```

## Usage

### Step 1: Train TStokenizer (Time Series Tokenizer)

First, train the VQ-VAE based time series tokenizer for each domain.

**Parameters for each dataset** (format: `d_model`, `n_embed`, `wave_length`):

| Dataset | d_model | n_embed | wave_length |
|---------|---------|---------|-------------|
| ECG (geo) | 64 | 128 | 40 |
| EEG (sleep) | 64 | 256 | 25 |
| FD (dev) | 64 | 512 | 40 |
| HAR | 64 | 256 | 1 |
| RWC (whale) | 64 | 384 | 32 |

```bash
cd TStokenizer

# Train tokenizer for HAR dataset
python main.py \
    --save_path ../vqvae/HAR \
    --dataset har \
    --data_path ../datasets/HAR \
    --device cuda:0 \
    --d_model 64 \
    --n_embed 256 \
    --wave_length 1

# Train tokenizer for EEG (sleep) dataset
python main.py \
    --save_path ../vqvae/EEG \
    --dataset sleep \
    --data_path ../datasets/EEG \
    --device cuda:0 \
    --d_model 64 \
    --n_embed 256 \
    --wave_length 25

# Train tokenizer for ECG (geo) dataset
python main.py \
    --save_path ../vqvae/ECG \
    --dataset geo \
    --data_path ../datasets/ECG \
    --device cuda:0 \
    --d_model 64 \
    --n_embed 128 \
    --wave_length 40

# Train tokenizer for FD (dev) dataset
python main.py \
    --save_path ../vqvae/FD \
    --dataset dev \
    --data_path ../datasets/FD \
    --device cuda:0 \
    --d_model 64 \
    --n_embed 512 \
    --wave_length 40

# Train tokenizer for RWC (whale) dataset
python main.py \
    --save_path ../vqvae/RWC \
    --dataset whale \
    --data_path ../datasets/RWC \
    --device cuda:0 \
    --d_model 64 \
    --n_embed 384 \
    --wave_length 32
```

### Step 2: Train InstructTime

#### Universal Training (Train from Scratch)

```bash
cd ..  # Back to project root

# Single domain training (e.g., HAR)
python run_truth_loss.py \
    --dataset har \
    --model_path ./gptmodel \
    --data_root ./datasets \
    --vqvae_root ./vqvae \
    --device cuda:0 \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5

# Multi-domain training
python run_truth_loss.py \
    --dataset mix \
    --model_path ./gptmodel \
    --data_root ./datasets \
    --vqvae_root ./vqvae \
    --device cuda:0 \
    --epochs 15 \
    --batch_size 16 \
    --lr 5e-5
```

#### Adaptation Training (Fine-tune from Pretrained)

```bash
python run_truth_loss.py \
    --dataset har \
    --model_path ./gptmodel/har \
    --load_model_path ./gptmodel/no_frozen/run_0/best_model \
    --data_root ./datasets \
    --vqvae_root ./vqvae \
    --device cuda:0 \
    --lr 1e-5 \
    --adapt
```

## Prompt Example

```
You will be receiving electroencephalogram(EEG) related signals.
Electroencephalogram signals: <BET><TS Tokens><EET>
The sleep patterns include waking up, rapid eye movement sleep, and sleep stages one through four, as well as periods of movement and unidentified stages.
Select one of the eight previously mentioned sleep patterns and report on the person's sleep using the provided information.
The person's sleep pattern is waking up
```

## Project Structure

```
InstructTime/
├── TStokenizer/          # Time Series Tokenizer (VQ-VAE)
│   ├── main.py           # Tokenizer training script
│   ├── model.py          # VQ-VAE model
│   └── ...
├── datasets/             # Dataset directory
├── vqvae/                # Trained tokenizer checkpoints
├── gpt2/                 # GPT-2 base model
├── run_truth_loss.py     # Main training script
├── multidataset.py       # Dataset processing
├── multimodel.py         # Model definition
├── args.py               # Argument parser
├── metrics.py            # Evaluation metrics
└── requirements.txt      # Dependencies
```

## License

This project is for research purposes only.
