import argparse

def get_hyperparams():
    parser = argparse.ArgumentParser(description="Input hyperparams.")

    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="./gptmodel", help="the path to save model.")
    parser.add_argument("--load_model_path", type=str, default="./gptmodel", help="the path to load pretrained model.")
    parser.add_argument('--dataset', type=str, default='mix', choices=['har', 'geo', 'sleep', 'mix', 'esr', 'ad', 'dev', 'whale'])

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--per_max_token", type=int, default=32, help="The maximum number of tokens for a label.")
    parser.add_argument("--encoder_max_length", type=int, default=230, help="Maximum length of language model input.")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--warm_up_ratio", type=float, default=0.05, help="Warm up step for schduler.")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs.")
    parser.add_argument("--adapt", type=bool, default=False, help="If finetune on pretrained model")

    parser.add_argument("--num_beams", type=int, default=1, help="Number of generation beams.")
    parser.add_argument("--num_return_sequences", type=int, default=1)

    args = parser.parse_args()
    return args