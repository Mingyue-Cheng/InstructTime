import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_normal_, constant_

"""
TCN based on RecBole's implementation
################################################

Reference code:
    - https://github.com/fajieyuan/nextitnet
    - https://github.com/initlisk/nextitnet_pytorch

"""

class TCN(nn.Module):
    def __init__(self, args=None, **kwargs):
        super(TCN, self).__init__()

        # load parameters info
        if args is not None:
            d_model = args.d_model
            self.embedding_size = args.d_model
            self.residual_channels = args.d_model
            self.block_num = args.block_num
            self.dilations = args.dilations * self.block_num
            self.kernel_size = args.kernel_size
            self.enabel_res_parameter = args.enable_res_parameter
            self.dropout = args.dropout
            self.device = args.device
            self.data_shape = args.data_shape
        else:
            d_model = kwargs['d_model']
            self.embedding_size = kwargs['d_model']
            self.residual_channels = kwargs['d_model']
            self.block_num = kwargs['block_num']
            self.dilations = kwargs['dilations'] * self.block_num
            self.data_shape = kwargs['data_shape']
            self.kernel_size = 3
            self.enabel_res_parameter = 1
            self.dropout = 0.1

        self.max_len = self.data_shape[0]
        print(self.max_len)

        # residual blocks    dilations in blocks:[1,2,4,8,1,2,4,8,...]
        rb = [
            ResidualBlock_b(
                self.residual_channels, self.residual_channels, kernel_size=self.kernel_size, dilation=dilation,
                enable_res_parameter=self.enabel_res_parameter, dropout=self.dropout
            ) for dilation in self.dilations
        ]
        self.residual_blocks = nn.Sequential(*rb)

        # fully-connected layer
        # self.output = nn.Linear(self.residual_channels, self.num_class)
        self.output = nn.Linear(d_model, d_model)
        self.broadcast_head = nn.Linear(d_model, self.data_shape[1])

        # parameters initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            xavier_normal_(module.weight.data)
            if module.bias is not None:
                constant_(module.bias.data, 0.1)

    def forward(self, x):
        # Residual locks
        # x in shape of [(B*T)*L*D]
        dilate_outputs = self.residual_blocks(x)
        x = dilate_outputs
        return self.output(x)


class ResidualBlock_b(nn.Module):
    r"""
    Residual block (b) in the paper
    """

    def __init__(self, in_channel, out_channel, kernel_size=10, dilation=None, enable_res_parameter=False, dropout=0):
        super(ResidualBlock_b, self).__init__()

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(out_channel, eps=1e-8)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kernel_size), padding=0, dilation=dilation * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(out_channel, eps=1e-8)

        self.dilation = dilation
        self.kernel_size = kernel_size

        self.enable = enable_res_parameter
        self.a = nn.Parameter(torch.tensor(1e-8))

    def forward(self, x):  # x: [batch_size, seq_len, embed_size]
        x_pad = self.conv_pad(x, self.dilation)  # [batch_size, embed_size, 1, seq_len+(self.kernel_size-1)*dilations]
        out = self.dropout1(self.conv1(x_pad).squeeze(2).permute(0, 2, 1))
        # [batch_size, seq_len+(self.kernel_size-1)*dilations-kernel_size+1, embed_size]
        out = F.relu(self.ln1(out))
        out_pad = self.conv_pad(out, self.dilation * 2)
        out2 = self.dropout2(self.conv2(out_pad).squeeze(2).permute(0, 2, 1))
        out2 = F.relu(self.ln2(out2))

        if self.enable:
            x = self.a * out2 + x
        else:
            x = out2 + x

        return x
        # return self.skipconnect(x, self.ffn)

    def conv_pad(self, x, dilation):
        """ Dropout-mask: To avoid the future information leakage problem, this paper proposed a masking-based dropout
        trick for the 1D dilated convolution to prevent the network from seeing the future items.
        Also the One-dimensional transformation is completed in this function.
        """
        inputs_pad = x.permute(0, 2, 1)
        inputs_pad = inputs_pad.unsqueeze(2)
        pad = nn.ZeroPad2d(((self.kernel_size - 1) * dilation, 0, 0, 0))
        inputs_pad = pad(inputs_pad)
        return inputs_pad
    
# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch

class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, beta=0.25):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps
        self.beta = beta

        embed = torch.randn(dim, n_embed)
        torch.nn.init.kaiming_uniform_(embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embed
                + self.embed.pow(2).sum(0, keepdim=True)
        )
        _, embed_ind = (-dist).max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            self.cluster_size.data.mul_(self.decay).add_(
                embed_onehot_sum, alpha=1 - self.decay
            )
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        commit_loss = (quantize - input.detach()).pow(2).mean()
        diff += commit_loss * self.beta
        quantize = input + (quantize - input).detach()

        return quantize, diff, embed_ind  # new_x, mse with input, index

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))

class Encoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()
        self.input_projection = nn.Linear(feat_num, hidden_dim)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        return self.blocks(self.input_projection(input))


class Decoder(nn.Module):
    def __init__(self, feat_num, hidden_dim, block_num, data_shape, dilations=[1, 4]):
        super().__init__()
        self.output_projection = nn.Linear(hidden_dim, feat_num)
        self.blocks = TCN(args=None, d_model=hidden_dim, block_num=block_num, data_shape=data_shape,
                          dilations=dilations)

    def forward(self, input):
        return self.output_projection(self.blocks(input))


class TStokenizer(nn.Module):
    def __init__(
            self,
            data_shape=(5000, 12),
            hidden_dim=64,
            n_embed=1024,
            block_num=4,
            wave_length=32,
    ):
        super().__init__()

        self.enc = Encoder(data_shape[1], hidden_dim, block_num, data_shape)
        self.wave_patch = (wave_length, hidden_dim)
        self.quantize_input = nn.Conv2d(1, hidden_dim, kernel_size=self.wave_patch, stride=self.wave_patch)
        self.quantize = Quantize(hidden_dim, n_embed)
        self.quantize_output = nn.Conv1d(int(data_shape[0] / wave_length), data_shape[0], kernel_size=1)
        self.dec = Decoder(data_shape[1], hidden_dim, block_num, data_shape)
        self.n_embed = n_embed
        self.hidden_dim = hidden_dim

    def get_name(self):
        return 'vqvae'

    def forward(self, input):
        enc = self.enc(input)
        enc = enc.unsqueeze(1)
        quant = self.quantize_input(enc).squeeze(-1).transpose(1, 2)
        quant, diff, id = self.quantize(quant)
        quant = self.quantize_output(quant)  # 2*100*64 -> 2*5000*64
        dec = self.dec(quant)
        # codes above are need, i.e. return dec_t, diff_t here will form our forward function

        return dec, diff, id
    
    def get_embedding(self, id):
        return self.quantize.embed_code(id)
    
    def decode_ids(self, id):
        quant = self.get_embedding(id)
        quant = self.quantize_output(quant)  # 2*100*64 -> 2*5000*64
        dec = self.dec(quant)

        return dec

if __name__ == '__main__':
    model = VQVAE()
    a = torch.randn(2, 5000, 8)
    tmp = model(a)
    print(1)
