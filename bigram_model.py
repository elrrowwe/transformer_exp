import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.model_selection import train_test_split

#TODO: read up on Bag of Words

#defining some hyperparameters
batch_size = 32
block_size = 8
train_iters = 3000
eval_interval = 300
lr = 0.01
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

oliver_twist = open('pg730.txt', 'r').read()

#getting all the characters, like in the n-gram model 
chars = sorted(list(set(oliver_twist)))
vocab_size = len(chars)

#a simple tokenizer
stoi = {s:i for i,s in enumerate(chars)}
itos = {i:s for s,i in stoi.items()}

encode = lambda s: [stoi[ch] for ch in s] #tokenize some characters
decode = lambda i: ' '.join([itos[num] for num in i]) #detokenize some integers

#tokenizing the entire data set
enc = torch.tensor(encode(oliver_twist), dtype=torch.long)

#splitting the text into train, test portions
train, test = train_test_split(enc, test_size = 0.2)


#a function which returns the 4 random sequences of context length 8
def get_batch(split):
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix]) #stacking the 1D tensors as rows into a matrix
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) #same thing for ys
    
    return x, y

@torch.no_grad()
def estimate_loss():
    #the function outputs the average loss over the train, test splits
    out = {} #the output placeholder dictionary 
    bi.eval() #setting the model to evaluation mode (a bigram model with just nn.Embedding will behave the same in both eval and train modes)
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters) #a placeholder tensor for split loss values

        for i in range(eval_iters): 
            X, y = get_batch(split)
            logits, loss = bi(X, y) #not really using the logits variable
            losses[i] = loss.item() #since loss is a tensor 

        out[split] = losses.mean() #averaging the losses over the 'split' split
    bi.train() #setting the model to trian mode again
    return out 

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        #creating a lookup table, just like in makemore #1
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        #idx, targets -- (B,T) tensors of integers                                 #Time = block_size
        logits = self.token_embedding_table(idx) # (a tensor of dimensions (Batch x Time x Channel))
        #logits -- the logs of probabilities of the characters being the next ones after some other characters
        
        if targets is None:
            loss = None
        else:
        
            #reshaping the tensor becaus of torch specifics 
            B, T, C = logits.shape
            logits = logits.view(B*T, C)

            targets = targets.view(B*T)

            #cross entropy = negative log likelihood
            loss = F.cross_entropy(logits, targets)

        return logits, loss
                      #idx - the current context of some characters in the current batch 
    def generate(self, idx, max_new_tokens):
        #idx - a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #getting the predictions 
            logits, loss = self(idx)
            #focus only on the last time step (pluck out the last value in the Time dimension, pytorch notation)
            logits = logits[:, -1, :] #transforms into (B, C)
            #apply the softmaax activation to probabilities
            probs = F.softmax(logits, dim=-1) #(B,C)
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) #(B,1)
            #append the sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)#
            
        return idx
    
bi = BigramModel(vocab_size)

optimizer = torch.optim.Adam(bi.parameters(), lr=lr)

#training the model
for i in range(train_iters):

    if i % eval_interval == 0:
        losses = estimate_loss()
        print(f'step: {i}, train_loss: {losses["train"]}, test loss: {losses["test"]}')

    #sampling new data
    xt, yt = get_batch('train')
    #calculating the loss
    logits, loss = bi(xt, yt)
    #zeroing all the gradients from the previous step
    optimizer.zero_grad(set_to_none=True)
    #getting the gradients of all the parameters
    loss.backward()
    #using the gradients to update the parameters
    optimizer.step()

idx = torch.zeros((1,1), dtype=torch.long) #explained in the notebook 
    
print(decode(bi.generate(idx, max_new_tokens=300)[0].tolist()))