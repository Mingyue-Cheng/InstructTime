import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

local_model_path = "./gpt2-model"
local_tokenizer_path = "./gpt2-tokenizer"

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(MLP, self).__init__()

        all_dims = [input_dim] + hidden_dims + [output_dim]

        self.linear_layers = nn.ModuleList()
        for i in range(len(all_dims) - 1):
            self.linear_layers.append(nn.Linear(all_dims[i], all_dims[i + 1]))

    def forward(self, x):
        for i, layer in enumerate(self.linear_layers):
            x = layer(x)
            if i < len(self.linear_layers) - 1:
                x = F.gelu(x)
        return x
    
class InstructTime(GPT2LMHeadModel):
    def __init__(self, config, ecgTokenizers, text_embedding=50258):
        super().__init__(config)
        self.ecgTokenizers = ecgTokenizers

        embed_vector = torch.empty(0, self.ecgTokenizers[0].hidden_dim)
        for tokenizer in self.ecgTokenizers:
            tokenizer_embed_vector = copy.deepcopy(tokenizer.quantize.embed).transpose(-1, 0)
            embed_vector = torch.cat([embed_vector, tokenizer_embed_vector], dim=0)
        self.embed_layer = nn.Embedding.from_pretrained(embed_vector)   
        
        self.text_embedding = text_embedding
        self.embed = config.n_embd
        self.config.pad_token_id = self.config.eos_token_id if self.config.pad_token_id is None else self.config.pad_token_id

        self.projection_layers = nn.ModuleList()
        for _ in ecgTokenizers:
            mlp = MLP(self.ecgTokenizers[0].hidden_dim, [64, 128, 256, 512], self.embed)
            mlp.apply(self.init_weights_kaiming)
            self.projection_layers.append(mlp)

        self.offsets = [self.text_embedding]
        for tokenizer in self.ecgTokenizers:
            self.offsets.append(self.offsets[-1] + tokenizer.n_embed)

    @staticmethod
    def init_weights_kaiming(m):
        if type(m) == nn.Linear:
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            m.bias.data.fill_(0.01)

    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"]

        text_mask = torch.lt(input_ids, self.text_embedding)
        ecg_mask = ~text_mask
        
        text_ids = input_ids.clone()
        text_ids[ecg_mask] = self.config.pad_token_id
        
        text_embeddings = self.transformer.wte(text_ids)
        text_embeddings.mul_(text_mask.float().unsqueeze(-1))

        masked_ids = input_ids.clone()
        masked_ids[text_mask] = 0
        masked_ids[ecg_mask] -= self.text_embedding

        ecg_embeddings = torch.zeros_like(text_embeddings)
        for i, _ in enumerate(self.ecgTokenizers):
            tokenizer_mask = (input_ids >= self.offsets[i]) & (input_ids < self.offsets[i + 1])
            tokenizer_ids = input_ids.clone()
            tokenizer_ids[~tokenizer_mask] = 0
            tokenizer_ids[tokenizer_mask] -= self.offsets[i]

            tokenizer_embeddings = self.embed_layer(tokenizer_ids)
            tokenizer_embeddings = self.projection_layers[i](tokenizer_embeddings)
            tokenizer_embeddings.mul_(tokenizer_mask.float().unsqueeze(-1))
            ecg_embeddings.add_(tokenizer_embeddings)

        kwargs["input_ids"] = None
        kwargs["inputs_embeds"] = ecg_embeddings + text_embeddings 
        
        outputs = super().forward(*args, **kwargs)
        return outputs    

class MultiTokenizer:
    def __init__(self, ecgTokenizers) -> None:
        self.textTokenizer = GPT2Tokenizer.from_pretrained(local_tokenizer_path)
        new_special_tokens = ["<BET>", "<EET>"]
        self.textTokenizer.add_special_tokens({"additional_special_tokens": new_special_tokens})
        self.text_vocab_size = len(self.textTokenizer)

        self.ecgTokenizers = ecgTokenizers

        self.pad_token_id = self.textTokenizer.eos_token_id
        self.eos_token_id = self.textTokenizer.eos_token_id

        self.offsets = self._calculate_offsets()

    def _calculate_offsets(self):
        offsets = []
        current_offset = self.text_vocab_size
        for tokenizer in self.ecgTokenizers:
            offsets.append(current_offset)
            current_offset += tokenizer.n_embed
        return offsets

    def vocabSize_all(self):
        return self.text_vocab_size + sum(tokenizer.n_embed for tokenizer in self.ecgTokenizers)

    def encode(self, input, model_id=1):
        if isinstance(input, str):
            return self.textTokenizer(input)["input_ids"]
        elif isinstance(input, torch.Tensor):
            input = input.to('cpu')
            if model_id < len(self.ecgTokenizers):
                tokenizer_index = model_id
                _, _, indices = self.ecgTokenizers[tokenizer_index](input)
                return indices + self.offsets[tokenizer_index]
            else:
                raise ValueError(f"Invalid model_id. Please provide a number between 0 and {len(self.ecgTokenizers)}.")
        else:
            raise ValueError("Unsupported input type. Please provide either a string or a torch.Tensor.")
        
    def decode(self, input, skip_special_tokens=True):        
        return self.textTokenizer.decode(input, skip_special_tokens=skip_special_tokens)