import torch
from torch.utils.data import Dataset
    
class MultiDataset(Dataset):
    r"""
    A Dataset Class for building Dataloader of ECG or other datasets.
    """

    def __init__(
        self,
        samples,
        tokenizer,
        mode: str,
        multi: str,
        encoder_max_length=256,
        prefix_text="",
    ) -> None:
        assert mode in ["train", "test"]
        super().__init__()
        self.samples = samples
        self.tokenizer = tokenizer
        self.mode = mode
        self.max_length = encoder_max_length
        self.multi = multi
        self.prefix_tokens = self.tokenizer.encode(prefix_text) if prefix_text else []
    
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
        label_ids = self.tokenizer.encode(label)

        if self.mode == "train":
            text = text + label
        else:
            text = text

        input_ids = self.template(ecg * 2.5, text)
        label_ids = [-100] * (len(input_ids) - len(label_ids)) + label_ids
        
        attn_masks = [1] * len(input_ids)
        input_ids, attn_masks = self.padding(input_ids, attn_masks)
        label_ids, _ = self.padding(label_ids, attn_masks)

        if self.mode == "train":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label_ids": torch.LongTensor(label_ids), 
            }

        elif self.mode == "test":
            return {
                "input_ids": torch.LongTensor(input_ids),
                "attn_masks": torch.FloatTensor(attn_masks),
                "label": label,
            }
        
    def template(self, ecg, text):
        r"""
        The contents of the items are stitched together according to a template to construct the input.
        """
        input_ids = self.prefix_tokens.copy()
        if self.multi == 'mix':
            if ecg.shape == (5000, 12):
                bet_ids = self.tokenizer.encode('Electrocardiogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=0)
            elif ecg.shape == (3000, 2): 
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=1)
            elif ecg.shape == (5120, 1):
                bet_ids = self.tokenizer.encode('Industrial equipment signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=2)
            elif ecg.shape == (128, 9):
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=3)
            elif ecg.shape == (4000, 1):
                bet_ids = self.tokenizer.encode('Whale sound signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=4)
            elif ecg.shape == (93, 13):
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
                ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0), model_id=5)
        else:
            if self.multi == 'geo':
                bet_ids = self.tokenizer.encode('Electrocardiogram signals: <BET>')
            elif self.multi == 'sleep':
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
            elif self.multi == 'esr':
                bet_ids = self.tokenizer.encode('Electroencephalogram signals: <BET>')
            else:
                bet_ids = self.tokenizer.encode('Human physical activities signals: <BET>')
            ecg_ids = self.tokenizer.encode(torch.Tensor(ecg).unsqueeze(0))
        text_ids = self.tokenizer.encode('<EET> \n' + text)

        ecg_ids = ecg_ids.tolist()
        ecg_ids = ecg_ids[0]

        input_ids.extend(bet_ids + ecg_ids + text_ids)

        if len(input_ids) > self.max_length:
            input_ids = input_ids[0 : self.max_length]
        
        return input_ids

    def padding(self, input_ids: list, attn_masks: list):
        r"""
        Padding the inputs for GPT model.

        For training, we pad the right side,
        For testing, we pad the left side.
        """
        assert len(input_ids) <= self.max_length

        if self.mode == "train":
            input_ids = input_ids + [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            )
            attn_masks = attn_masks + [0] * (self.max_length - len(attn_masks))
        elif self.mode == "dev" or self.mode == "test":
            input_ids = [self.tokenizer.pad_token_id] * (
                self.max_length - len(input_ids)
            ) + input_ids
            attn_masks = [0] * (self.max_length - len(attn_masks)) + attn_masks
        return input_ids, attn_masks