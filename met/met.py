import inspect
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gen_all_binary_vectors(length):
    return ((torch.arange(2 ** length).unsqueeze(1) >> torch.arange(length - 1, -1, -1)) & 1).float()


def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input_data):
        return F.layer_norm(input_data, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """
    masked multi-head self-attention part
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.emb_dim = kwargs["embedding_dimension"]
        arg_bias = kwargs["bias"]
        self.dropout_rate = kwargs["dropout_rate"]
        self.n_head = kwargs["n_head"]
        self.N = kwargs["num_species"]
        self.M = kwargs["num_reactions"]
        self.block_size = self.N * 2 + self.M + 1  # tokens, init, params, time, state
        assert self.emb_dim % self.n_head == 0, f"embedding dimension {self.emb_dim} is not consistent with number of attention heads {self.n_head}"
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(self.emb_dim, 3 * self.emb_dim, bias=arg_bias)
        # output projection
        self.c_proj = nn.Linear(self.emb_dim, self.emb_dim, bias=arg_bias)
        # regularization
        self.attn_dropout = nn.Dropout(self.dropout_rate)
        self.resid_dropout = nn.Dropout(self.dropout_rate)
        # flash attention, support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(self.block_size, self.block_size))
                                 .view(1, 1, self.block_size, self.block_size))

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.emb_dim, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                                                 dropout_p=self.dropout_rate if self.training else 0,
                                                                 is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    """
    multi-layer perceptor
    input and output: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, **kwargs):
        super().__init__()
        emb_dim = kwargs["embedding_dimension"]
        bias = kwargs["bias"]
        dropout_rate = kwargs["dropout_rate"]
        self.c_fc = nn.Linear(emb_dim, 4 * emb_dim, bias=bias)
        self.c_proj = nn.Linear(4 * emb_dim, emb_dim, bias=bias)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        emb_dim = kwargs["embedding_dimension"]
        bias = kwargs["bias"]
        self.ln_1 = LayerNorm(emb_dim, bias=bias)
        self.attn = CausalSelfAttention(**kwargs)
        self.ln_2 = LayerNorm(emb_dim, bias=bias)
        self.mlp = MLP(**kwargs)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class MET(nn.Module):
    """
    transformer for estimating joint distribution of cme
    """

    def __init__(self, *args, **kwargs):
        super(MET, self).__init__()
        self.N = kwargs["num_species"]
        self.M = kwargs["num_reactions"]
        # use a fixed prompt input, useful for combining models with different sizes
        self.var_dim = kwargs["variable_dimension"]  # dimension of the input prompt after perceptron
        self.block_size = kwargs["block_size"]  # tokens, consists of at least (params, init, time)
        assert self.block_size >= self.var_dim + self.N, f"block size {self.block_size} is smaller than input size {self.var_dim+self.N}"
        # assign a larger block_size is useful if model reaction matrices are also imported
        # each chuck of information can be divided by separators
        self.U = kwargs["state_upper_bound"]  # vocab_size in GPT, also the number of discrete for params and time
        self.bits = kwargs["bits"]
        self.bias = kwargs["bias"]  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        self.dropout_rate = kwargs["dropout_rate"]
        self.emb_dim = kwargs["embedding_dimension"]
        self.ff_dim = kwargs["feed_forward_dimension"]
        self.n_layer = kwargs["num_encoder_layers"]
        self.n_head = kwargs["n_head"]
        self.device = kwargs["device"]
        self.constrains = kwargs["constrains"]

        self.pro_fc = nn.Linear(self.M+self.N+1, self.var_dim)
        self.pro_emb = nn.Linear(1, self.emb_dim, bias=False)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(self.U, self.emb_dim),  # weighted target embedding
            wpe=nn.Embedding(self.block_size, self.emb_dim),  # weighted positional embedding
            drop=nn.Dropout(self.dropout_rate),
            h=nn.ModuleList([Block(**kwargs) for _ in range(self.n_layer)]),
            ln_f=LayerNorm(self.emb_dim, bias=self.bias),
        ))

        self.lm_head = nn.Linear(self.emb_dim, self.U, bias=False)  # language model head

        self.transformer.wte.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # allowing different length of prompts
        # useful for generating new models and configurations, then obtain states at also a generated time
        device = idx.device
        b, t = idx.size()  # batch_size, tokens
        assert t <= self.block_size, f"Cannot forward sequence of length {t+1} (with start symbol), block size is {self.block_size}"

        ps = self.M + self.N + 1
        pro = idx[:, :ps]
        pro = self.pro_fc(pro)
        pro = new_gelu(pro)
        pro = pro.unsqueeze(-1)
        pro = self.pro_emb(pro)

        state = idx[:, ps:]
        state = self.transformer.wte(state.int())

        tok_emb = torch.cat((pro, state), dim=1)
        pos = torch.arange(0, tok_emb.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        x = self.lm_head(x)

        # compute conditional probabilities
        if self.constrains[0] == 0 or np.all(self.constrains == self.U):
            lcp = F.log_softmax(x, dim=2)
            # lcp = log p (pro, tgt) = log p (tgt | pro) + log p(pro)
            # p(pro) = 1
        else:
            lcp = torch.ones_like(x) * (-100)
            index1 = np.arange(self.N)[self.constrains < self.U]
            index0 = np.arange(self.N)[self.constrains == self.U]
            index11 = self.constrains[index1[0]]
            index00 = self.constrains[index0[0]]
            # since prompt has fixed input length, the index should exclude prompt inputs
            index1 += self.var_dim-1  # indexes of the species that have constrains smaller than U
            index0 += self.var_dim-1  # indexes of the species that have constrains equal to U
            index1 = index1[index1 < x.shape[1]]
            index0 = index0[index0 < x.shape[1]]
            lcp[:, index0, :index00] = F.log_softmax(x[:, index0, :index00], dim=2)
            lcp[:, index1, :index11] = F.log_softmax(x[:, index1, :index11], dim=2)

        return lcp

    def log_joint_prob(self, idx):
        # assert idx.shape[1] == self.block_size, "joint probability can only be calculated for full sequence"
        b, t = idx.shape
        device = idx.device
        ljp = torch.zeros(b, self.N, device=device)
        for i in range(self.N):
            tmp = idx[:, :-self.N+i]
            lcp = self(tmp)
            ids = idx[:, -self.N+i].long()
            ljp[:, i] = lcp[:, -1, :].gather(1, ids.view(-1, 1))[:, 0]

        ljp_sum = ljp.sum(dim=1)  # log_joint_prob is normalized naturally within each token

        return ljp_sum

    @torch.no_grad()
    def sample(self, idx):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        assert idx.shape[1] == self.N + self.M + 1, "current prompts only contain (params, init, time)"
        samples = torch.zeros(idx.shape[0], self.N, device=self.device)
        for i in range(self.N):
            lcp = self(idx)
            # if all number constrains equals U
            if self.constrains[0] == 0 or np.all(self.constrains == self.U):
                cp = torch.exp(lcp[:, -1, :])
                idx_next = torch.multinomial(cp, num_samples=1)
                samples[:, i] = idx_next[:, 0]
                idx = torch.cat((idx, idx_next), dim=1)
            # if constrains is specified for each species
            else:
                cp = torch.exp(lcp[:, -1, :self.constrains[i]])
                idx_next = torch.multinomial(cp, num_samples=1)
                samples[:, i] = idx_next[:, 0]
                idx = torch.cat((idx, idx_next), dim=1)

        return idx, samples

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


if __name__ == '__main__':
    kwargs_dict = {
        'num_species': 2,
        "num_reactions": 4,
        "variable_dimension": 32,
        "block_size": 64,
        'state_upper_bound': 2,
        'bits': 0,
        "bias": False,
        "dropout_rate": 0.0,
        'embedding_dimension': 64,
        'feed_forward_dimension': 128,
        'num_encoder_layers': 6,
        'n_head': 8,
        'device': "cpu",
        'constrains': [0],
        "batch_size": 4,
        "epsilon": 1e-10,
    }

    model = MET(**kwargs_dict)
    print(model)
    # # test normalization condition
    sample = gen_all_binary_vectors(kwargs_dict["num_species"])
    # s = np.array([1, 2, 3, 4, 5, 6, 0.1]).reshape(1, -1).repeat(4, axis=0)
    NP = 2  # number of prompts
    BS = 4
    prompt1 = np.array([1, 0, 1, 0, 0, 0, 1])
    prompt1 = prompt1.reshape(1, -1).repeat(BS, axis=0)
    prompt2 = np.array([0, 0, 0, 0, 0, 1, 1])
    prompt2 = prompt2.reshape(1, -1).repeat(BS, axis=0)
    prompt = np.concatenate((prompt1, prompt2), axis=0)
    prompt = torch.from_numpy(prompt)
    sample = sample.repeat(NP, 1).long()
    input_idx = torch.cat((prompt, sample), dim=1).long()
    predict_prob = model.log_joint_prob(input_idx.float())
    predict_prob = predict_prob.exp()
    for i_p in range(NP):
        print(predict_prob[i_p * BS: (i_p + 1) * BS].sum())  # the log_joint_prob is normalized for each token
    # test sampling
    props, samples = model.sample(prompt.float())
    print(props.shape, samples.shape)
