from torch.utils.data import Dataset

class LSTM_Dataset(Dataset):
    def __init__(self, sentences, mode, vocab=None):
        self.mode = mode
        self.sentences = sentences
        if mode == 'train':
            self.word_to_ix = self._create_vocab()
        else:
            self.word_to_ix = vocab

    def _create_vocab(self):
        vocab = {word: i for i, word in enumerate(set([token for sent in self.sentences for token in sent]))}
        vocab['unk'] = len(vocab)
        vocab['pad'] = len(vocab)
        return vocab

    def _get_word_index(self, word):
        return self.word_to_ix.get(word, self.word_to_ix['unk'])

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]    