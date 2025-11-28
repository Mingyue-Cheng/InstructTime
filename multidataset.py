import torch
from torch.utils.data import Dataset
    
class MultiDataset(Dataset):
    r"""
    A Dataset Class for building Dataloader of ECG or other datasets.
    Implementation kept aligned with the original instructtimeoriginv1.1 version.
    """

    def __init__(
        self,
        samples,
        tokenizer,
        mode: str,
        multi: str,
        encoder_max_length=256,
        prefix_text="",
        dataset_key=None,
    ) -> None:
        assert mode in ["train", "test"]
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = encoder_max_length
        self.multi = multi.lower()
        self.prefix_tokens = self.tokenizer.encode(prefix_text) if prefix_text else []
        # Store per-domain key to select the proper tokenizer
        alias_map = {'sleep': 'eeg', 'dev': 'fd', 'whale': 'rwc'}
        ds_key = (dataset_key if dataset_key else multi).lower()
        self.dataset_key = alias_map.get(ds_key, ds_key)
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, ecg, _ = self.samples[idx]

        dx_index = text.find("information.\n")
        if dx_index != -1:
            label = text[dx_index + 13:]
            text = text[:dx_index + 13]
        else:
            label = ''
        
        if self.mode == "train":
            # Training: input_ids = prompt + answer (+ EOS if enabled)
            text_with_label = text + label
            input_ids = self.template(ecg * 2.5, text_with_label, add_eos=True)
            
            # Standard causal LM style: labels = input_ids.clone() with prompt positions set to -100
            # Find where the label starts (length of label + EOS)
            label_with_eos_ids = self.tokenizer.encode(label) + [self.tokenizer.eos_token_id]
            label_len = len(label_with_eos_ids)
            prompt_len = len(input_ids) - label_len
            
            # labels follow input_ids but prompt positions are masked with -100
            label_ids = [-100] * prompt_len + input_ids[prompt_len:]
            
            return {
                "input_ids": torch.LongTensor(input_ids),
                "label_ids": torch.LongTensor(label_ids),
                "pad_token_id": self.tokenizer.pad_token_id,
            }

        elif self.mode == "test":
            # Evaluation: only prompt is fed, label is kept as plain text
            input_ids = self.template(ecg * 2.5, text, add_eos=False)
            
            return {
                "input_ids": torch.LongTensor(input_ids),
                "label": label,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
        
    def template(self, ecg, text, add_eos=False):
        r"""
        The contents of the items are stitched together according to a template to construct the input.
        add_eos: when True in training, append an EOS token at the end of the sequence.
        """
        input_ids = self.prefix_tokens.copy()
        prompt_map = {
            'geo': 'Electrocardiogram signals: <BET>',
            'eeg': 'Electroencephalogram signals: <BET>',
            'esr': 'Electroencephalogram signals: <BET>',
            'fd': 'Industrial equipment signals: <BET>',
            'har': 'Human physical activities signals: <BET>',
            'rwc': 'Whale sound signals: <BET>',
        }

        if self.multi == 'mix':
            shape_to_key = {
                (5000, 12): 'geo',
                (3000, 2): 'eeg',
                (5120, 1): 'fd',
                (128, 9): 'har',
                (4000, 1): 'rwc',
                (93, 13): 'eeg',
            }
            dataset_key = shape_to_key.get(ecg.shape)
            if dataset_key is None:
                raise ValueError(f"Unsupported signal shape {ecg.shape} for mix dataset.")
        else:
            dataset_key = self.dataset_key

        prompt = prompt_map.get(dataset_key, 'Human physical activities signals: <BET>')
        bet_ids = self.tokenizer.encode(prompt)
        model_id = self.tokenizer.get_model_index(dataset_key)
        ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=model_id)
        text_ids = self.tokenizer.encode('<EET> \n' + text)

        ecg_ids = ecg_ids.tolist()
        ecg_ids = ecg_ids[0]

        input_ids.extend(bet_ids + ecg_ids + text_ids)
        
        # Optionally append EOS during training
        if add_eos:
            input_ids.append(self.tokenizer.eos_token_id)

        # Dynamic padding: do not truncate here; let the collate function handle padding.
        # If the sequence is too long, optionally truncate while preserving the right side (label and EOS).
        if self.max_length and len(input_ids) > self.max_length:
            # Truncate from the left, preserving the right side containing label and EOS.
            input_ids = input_ids[-self.max_length:]
        
        return input_ids

    def padding(self, input_ids: list, attn_masks: list):
        r"""
        Padding the inputs for GPT model.
        Behavior consistent with the original:
        - Train: right-side padding (standard causal LM behavior).
        - Test: left-side padding (needed for generation).
        """
        assert len(input_ids) <= self.max_length

        if self.mode == "train":
            # Train: right-side padding
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        elif self.mode == "dev" or self.mode == "test":
            # Test/Dev: left-side padding
            input_ids = [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            ) + input_ids
            attn_masks = [0] * (self.max_length - len(attn_masks)) + attn_masks
        return input_ids, attn_masks
