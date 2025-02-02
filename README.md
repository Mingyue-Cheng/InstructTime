<div align="center">
  <!-- <h1><b> Time-LLM </b></h1> -->
  <!-- <h2><b> Time-LLM </b></h2> -->
  <h2><b> InstructTime: Advancing Time Series Classification with Multimodal Language Modeling (WSDM2025, Accepted) </b></h2>
</div>

---
>
> 🙋 Please let us know if you find out a mistake or have any suggestions!
> 📄 **[Read our paper on arXiv](https://arxiv.org/abs/2403.12371)**
> 🌟 If you find this resource helpful, please consider to star this repository and cite our research:

```
@article{cheng2024advancing,
  title={Advancing Time Series Classification with Multimodal Language Modeling},
  author={Cheng, Mingyue and Chen, Yiheng and Liu, Qi and Liu, Zhiding and Luo, Yucong},
  journal={arXiv preprint arXiv:2403.12371},
  year={2024}
}
```

## Project Overview

This is an anonymously open-sourced project designed for scientific research and technological development, supporting the blind review process to ensure fairness and objectivity in evaluations. 

## Installation Instructions

```bash
# Install dependencies
pip install -r requirements.txt

# Run to train TStokenizer
cd TStokenizer
python main.py \
--save_path $VQVAE_PATH \
--dataset $DATASET \
--data_path $DATA_PATH \
--device $DEVICE \
--d_model $D_MODEL \
--wave_length $WAVE_LENGTH \
--n_embed $NUM_TOKEN \

# Run to train Instructtime-Universal
python main.py \
--save_path $VQVAE_PATH \
--dataset $DATASET \
--model_path $DATA_PATH \
--device $DEVICE \
--adapt False

# Run to train Instructtime-Adapt
python main.py \
--save_path $VQVAE_PATH \
--dataset $DATASET \
--model_path $DATA_PATH \
--load_model_path $DATA_PATH \
--device $DEVICE \
--lr $lr \
--adapt True
```

## One of Instructime's Prompt

```
You will be receiving electroencephalogram(EEG) related signals.
EEG: <BET><TS Tokens><EET>
The sleep patterns include waking up, rapid eye movement sleep, and sleep stages one through four, as well as periods of movement and unidentified stages.
Select one of the eight previously mentioned sleep patterns and report on the person's sleep using the provided information.
The person's sleep pattern is waking up
```
