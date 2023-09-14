# %%
import torch
import preprocessing as pp
import lstm_dataset as ds
import lstm_model as mdl

# %% [markdown]
# ### Dataset Creation

# %%
from torch.utils.data import DataLoader
import random

# %%
training_set_length = 29000
testing_set_length = 20000
validation_set_length = 10000

random.seed(42)
random.shuffle(pp.sentences)

training_set = pp.sentences[:training_set_length]
testing_set = pp.sentences[training_set_length:training_set_length+testing_set_length]
validation_set = pp.sentences[training_set_length+testing_set_length:]

# %%
def collate_fn(batch):
    max_len = max([len(sent) for sent in batch])
    batch = [sent + ['pad'] * (max_len - len(sent)) for sent in batch]
    batch = [[word_to_ix.get(word, word_to_ix['unk']) for word in sent] for sent in batch]

    input_tensor = torch.tensor([sent[:-1] for sent in batch])
    target_truth = torch.tensor([sent[1:] for sent in batch])
    return input_tensor, target_truth

# %%
train_dataset = ds.LSTM_Dataset(training_set, 'train')

word_to_ix = train_dataset.word_to_ix

val_dataset = ds.LSTM_Dataset(validation_set, 'val', vocab=word_to_ix)
test_dataset = ds.LSTM_Dataset(testing_set, 'test', vocab=word_to_ix)

batch_size = 128

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# %% [markdown]
# ### Model Creation

# %%
import gensim
import gensim.downloader

glove_vectors = gensim.downloader.load('glove-wiki-gigaword-100')

# %%
vocab_size = len(word_to_ix)

embedding_matrix = torch.zeros((vocab_size, glove_vectors.vector_size))

for word, i in word_to_ix.items():
    if word in glove_vectors:
        embedding_matrix[i] = torch.tensor(glove_vectors[word])
    else:
        embedding_matrix[i] = torch.tensor(glove_vectors['unk'])
print(embedding_matrix.size())

model = mdl.LSTM_Model(embedding_matrix, vocab_size)


# %% [markdown]
# ### Training

# %%
def run_epoch(model, data_loader, loss_fn, optimizer=None):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0

    for input_tensor, target_truth in data_loader:

        input_tensor = input_tensor.cuda()
        target_truth = target_truth.cuda()

        model_hidden = None
        target_pred = []

        for i in range(input_tensor.size(1)):
            output_tensor, model_hidden = model(input_tensor[:, i].unsqueeze(1), model_hidden)
            target_pred.append(output_tensor)

        target_pred = torch.cat(target_pred, dim=1)
        loss = loss_fn(target_pred.view(-1, target_pred.size(-1)), target_truth.view(-1))
        total_loss += loss.item()

        if optimizer:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    return total_loss / len(data_loader)



# %%
import torch.nn as nn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

model.cuda()

best_val_loss = float('inf')

all_val_loss = []
all_train_loss = []


for epoch in range(num_epochs):
    train_loss = run_epoch(model, train_loader, loss_fn, optimizer)
    all_train_loss.append(train_loss)
    with torch.no_grad():
        val_loss = run_epoch(model, val_loader, loss_fn)
        all_val_loss.append(val_loss)

    print('Epoch: {}, Train Loss: {:.4f}, Val Loss: {:.4f}'.format(epoch+1, train_loss, val_loss))
    if val_loss < best_val_loss:
        counter = 0
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_lstm_model.pth')
    else:
        counter += 1
        if counter == 3:
            break

# %%
# import numpy as np

# def get_graph(train_loss, val_loss):
#     import seaborn as sns
#     import matplotlib.pyplot as plt
#     sns.set()
#     x = np.arange(1, len(train_loss) + 1)
#     plt.figure(figsize=(12, 6))
#     plt.title('Training/Validation Loss vs Epoch (LSTM)')
#     plt.plot(x, train_loss, label='Training Loss')
#     plt.plot(x, val_loss, label='Validation Loss')
#     plt.xticks(x)
#     plt.xlabel('Epoch')
#     plt.ylabel('Cross Entropy Loss')
#     plt.legend()
#     plt.show()

# get_graph(all_train_loss, all_val_loss)


# %% [markdown]
# ### Testing & Perplexity Score

# %%
def index_to_word(index):
  return list(word_to_ix.keys())[list(word_to_ix.values()).index(index)]

# %%
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

# %%
import numpy as np

loss_fn = nn.CrossEntropyLoss(ignore_index=word_to_ix['pad'])

def get_perplexity(model, data_set, file):

    model.load_state_dict(torch.load('best_lstm_model.pth'))
    model.eval()

    perplexity = []

    with torch.no_grad():
        for input_tensor, target_tensor in data_set:
            target_pred = []
            input_tensor = input_tensor.cuda()
            target_tensor = target_tensor.cuda()

            model_hidden = None
            for i in range(input_tensor.shape[1]):
                word_i = input_tensor[:, i].unsqueeze(1).cuda()
                i_output, model_hidden = model(word_i, model_hidden)
                target_pred.append(i_output)

            target_pred = torch.stack(target_pred, dim=1)
            loss = loss_fn(target_pred.view(-1, target_pred.size(-1)), target_tensor.view(-1))
            curr_perplexity = np.exp(loss.item())
            perplexity.append(curr_perplexity)

            sent = ' '.join([index_to_word(word) for word in input_tensor[0].tolist()])

            with open(file, 'a') as f:
                f.write('{} \t {}\n'.format(sent, curr_perplexity))

    avg_perplexity = np.mean(perplexity)

    with open(file, 'a') as f:
        f.write('Average Perplexity: {:.4f}\n'.format(avg_perplexity))

    return avg_perplexity

# %%
train_perplexity = get_perplexity(model, train_loader, '2020101121-LM2-train-perplexity.txt')
print('Train Perplexity: {:.4f}'.format(train_perplexity))

val_perplexity = get_perplexity(model, val_loader, '2020101121-LM2-val-perplexity.txt')
print('Val Perplexity: {:.4f}'.format(val_perplexity))

test_perplexity = get_perplexity(model, test_loader, '2020101121-LM2-test-perplexity.txt')
print('Test Perplexity: {:.4f}'.format(test_perplexity))


