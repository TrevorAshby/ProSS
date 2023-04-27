# An autoregressive transformer decoder


# masked multi-head attention
# add & norm with x
# multi-head attention
# add & norm with x+1
# ff
# add & norm

import torch
import math
import copy
import time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn
from torch.autograd import Variable
import matplotlib.pyplot as plt

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))    

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N, V):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        self.generator = Generator(layer.size, V)
        
    #def forward(self, x, memory, src_mask, tgt_mask):
    def forward(self, x, tgt_mask):
        for layer in self.layers:
            #x = layer(x, memory, src_mask, tgt_mask)
            x = layer(x, tgt_mask)
        return self.norm(x)
    
class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2) # was 3
 
    #def forward(self, x, memory, src_mask, tgt_mask):
    def forward(self, x, tgt_mask):
        "Follow Figure 1 (right) for connections."
        #m = memory
       
        #print(tgt_mask)
        #print(tgt_mask.shape)
        x = self.sublayer[0](x, lambda x: self.src_attn(x, x, x, tgt_mask))
        #x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[1](x, self.feed_forward) # was 2
    
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)
    

if __name__ == "__main__":
    class Batch:
        "Object for holding a batch of data with mask during training."
        def __init__(self, src, trg=None, pad=0):
            self.src = src
            self.src_mask = (src != pad).unsqueeze(-2)
            if trg is not None:
                self.trg = trg[:,:]#trg[:, :-1]
                self.trg_y = trg[:, 1:]
                self.trg_mask = \
                    self.make_std_mask(self.trg, pad)
                self.ntokens = (self.trg_y != pad).data.sum()
        
        @staticmethod
        def make_std_mask(tgt, pad):
            "Create a mask to hide padding and future words."
            tgt_mask = (tgt != pad).unsqueeze(-2)
            tgt_mask = tgt_mask & Variable(
                subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
            return tgt_mask
        
    def data_gen(V, batch, nbatches, lines):
        "Generate random data for a src-tgt copy task."
        for i in range(nbatches):
            #data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
            
            from dataset import sentence2vec
            #vec = sentence2vec("<BOS> how are you <EOS>")
            vec = sentence2vec(lines[i])
            #vec = position(encode(torch.tensor(vec)))


            data = torch.from_numpy(np.array(vec)).unsqueeze(0)
            #data[:, 0] = 1
            src = Variable(data, requires_grad=False)
            tgt = Variable(data, requires_grad=False)
            yield Batch(src, tgt, 0)

    d_model = 256
    N = 4
    h = 8
    dropout = 0.1
    d_ff = 2048
    V = 8983 + 3

    n_lines = 13183
    lines = open('./tfotr_text_lines.txt', 'r').readlines()

    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    decoder = Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N, V)

   
    encode = Embeddings(d_model, V)
    
    from dataset import sentence2vec
    vec = sentence2vec("<BOS> how are you <EOS>")
    vec = encode(torch.tensor(vec))
    print(vec)
    

    # data_iter = data_gen(V, 30, 20)
    # for i, batch in enumerate(data_iter):

    #     out = decoder.forward(position(encode(batch.src)), batch.src_mask)
    #     break
    
    # print(out.shape)

    # out = decoder.generator(out)
    # print(out.shape)
    # print(out[0,1,:])






    def run_epoch(data_iter, dec, loss_compute):
        "Standard Training and Logging Function"
        encode = Embeddings(256, 8983 + 3)
        start = time.time()
        total_tokens = 0
        total_loss = 0
        tokens = 0
        for i, batch in enumerate(data_iter):
            out = dec.forward(position(encode(batch.src)), batch.trg_mask)
            loss = loss_compute(out, batch.trg_y, batch.ntokens)
            total_loss += loss
            total_tokens += batch.ntokens
            tokens += batch.ntokens
            if i % 50 == 1:
                elapsed = time.time() - start
                print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                        (i, loss / batch.ntokens, tokens / elapsed))
                start = time.time()
                tokens = 0
        return total_loss / total_tokens


    class NoamOpt:
        "Optim wrapper that implements rate."
        def __init__(self, model_size, factor, warmup, optimizer):
            self.optimizer = optimizer
            self._step = 0
            self.warmup = warmup
            self.factor = factor
            self.model_size = model_size
            self._rate = 0
            
        def step(self):
            "Update parameters and rate"
            self._step += 1
            rate = self.rate()
            for p in self.optimizer.param_groups:
                p['lr'] = rate
            self._rate = rate
            self.optimizer.step()
            
        def rate(self, step = None):
            "Implement `lrate` above"
            if step is None:
                step = self._step
            return self.factor * \
                (self.model_size ** (-0.5) *
                min(step ** (-0.5), step * self.warmup ** (-1.5)))
            
    def get_std_opt(model):
        return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))


    class LabelSmoothing(nn.Module):
        "Implement label smoothing."
        def __init__(self, size, padding_idx, smoothing=0.0):
            super(LabelSmoothing, self).__init__()
            self.criterion = nn.KLDivLoss(size_average=False)
            self.padding_idx = padding_idx
            self.confidence = 1.0 - smoothing
            self.smoothing = smoothing
            self.size = size
            self.true_dist = None
            
        def forward(self, x, target):
            assert x.size(1) == self.size
            true_dist = x.data.clone()
            true_dist.fill_(self.smoothing / (self.size - 2))
            true_dist.scatter_(1, target.data.unsqueeze(1).type(torch.int64), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target.data == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)
            self.true_dist = true_dist
            return self.criterion(x, Variable(true_dist, requires_grad=False))

    class SimpleLossCompute:
        "A simple loss compute and train function."
        def __init__(self, generator, criterion, opt=None):
            self.generator = generator
            self.criterion = criterion
            self.opt = opt
            
        def __call__(self, x, y, norm):
            x = self.generator(x)
            loss = self.criterion(x.contiguous().view(-1, x.size(-1)), 
                                y.contiguous().view(-1)) / norm
            loss.backward()
            if self.opt is not None:
                self.opt.step()
                self.opt.optimizer.zero_grad()
                #print('loss.data: ', loss.data)
            #return loss.data[0] * norm
            return loss.data.item() * norm

    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)

    model_opt = NoamOpt(d_model, 1, 400,
            torch.optim.Adam(decoder.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    for epoch in range(20):
        decoder.train()
        run_epoch(data_gen(V, 30, n_lines, lines), decoder, 
                SimpleLossCompute(decoder.generator, criterion, model_opt))
        decoder.eval()
        print(run_epoch(data_gen(V, 30, 5, lines), decoder, 
                        SimpleLossCompute(decoder.generator, criterion, None)))
        

    pytorch_total_params = sum(p.numel() for p in decoder.parameters())
    print(pytorch_total_params)