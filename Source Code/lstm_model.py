import torch
import torch.nn as nn

class LSTM_Model(nn.Module):
    def __init__(self, embedding_matrix, vocab_size):
        super(LSTM_Model, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
        self.lstm = nn.LSTM(embedding_matrix.shape[1], 300, batch_first=True)
        self.linear = nn.Linear(300, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embedding(x)
        x, hidden = self.lstm(x, hidden)
        x = self.linear(x)
        return x, hidden