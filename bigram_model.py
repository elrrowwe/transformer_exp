import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

# TODO: read up on Bag of Words
# TODO: explain dropout (paper)
# TODO: explain projection

# temporarily added THE HAUNTED MAN AND THE GHOST’S BARGAIN to the oliver_twist data set

# defining some hyperparameters
batch_size = 16  # how many sequences (sentences of some length) are processed at once
block_size = 32  # how long the aforementioned sequences are (context length)
train_iters = 5000
eval_interval = 100
lr = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
eval_iters = 200
n_embed = 64  # the number of embedding dimensions (the dimensionality of embedding vectors)
n_head = 4  # the number of heads in the multi-head attention layer
n_layer = 6
dropout = 0.0  # the dropout rate

# oliver_twist = open('pg73'0.txt', 'r', encoding='utf8').read()
shakespeare = open('tiny_shakespeare.txt', 'r', encoding='utf8').read()

# getting all the characters, like in the n-gram model
chars = sorted(list(set(shakespeare)))
vocab_size = len(chars)

# a simple (de-)tokenizer
stoi = {s: i for i,s in enumerate(chars)}
itos = {i: s for i,s in enumerate(chars)}

encode = lambda s: [stoi[ch] for ch in s]  # tokenize some characters
decode = lambda i: ''.join([itos[num] for num in i])  # detokenize some integers

# tokenizing the entire data set
enc = torch.tensor(encode(shakespeare), dtype=torch.long, device=device)

# splitting the text into train, test portions
train, test = train_test_split(enc, shuffle=False, test_size=0.1)


def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,), device=device)
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}  # the output placeholder dictionary
    bi.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters, device=device)

        for i in range(eval_iters):
            X, y = get_batch(split)
            logits, loss = bi(X, y)
            losses[i] = loss.item()

        out[split] = losses.mean()
    bi.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False).to(device)
        self.query = nn.Linear(n_embed, head_size, bias=False).to(device)
        self.value = nn.Linear(n_embed, head_size, bias=False).to(device)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)).to(device))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed).to(device)
        self.dropout = nn.Dropout(dropout).to(device)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        
        return out


class Dense(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed).to(device),
            nn.ReLU().to(device),
            nn.Linear(4 * n_embed, n_embed).to(device),
            nn.Dropout(dropout).to(device)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.attn = MultiHeadAttention(n_head, head_size)
        self.ffn = Dense(n_embed)
        self.ln1 = nn.LayerNorm(n_embed).to(device)
        self.ln2 = nn.LayerNorm(n_embed).to(device)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))

        return x


class BigramModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed).to(device)
        self.position_embedding_table = nn.Embedding(block_size, n_embed).to(device)
        self.lm_head = nn.Linear(n_embed, vocab_size).to(device)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed).to(device)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_embeddings = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embeddings + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


bi = BigramModel().to(device)

optimizer = torch.optim.AdamW(bi.parameters(), lr=lr)

for i in range(train_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'step: {i}, train_loss: {losses["train"]}, test loss: {losses["test"]}')

    xt, yt = get_batch('train')
    logits, loss = bi(xt, yt)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

idx = torch.zeros((1, 1), dtype=torch.long, device=device)

print(decode(bi.generate(idx, max_new_tokens=2000)[0].tolist()))