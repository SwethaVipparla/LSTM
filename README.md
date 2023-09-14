# LSTM Model

LSTM (Long Short Term Memory) is an advanced version of recurrent neural network (RNN) architecture that was designed to model chronological sequences and their long-range dependencies more precisely than conventional RNNs. 

The entire code is split into 4 files, which have different purposes.

- `preprocessing.py`
  Contains the code for preprocessing the data. The unicode characters are normalised, repeating characters are removed, and the important punctuations are retained.
  In addition, `sos` and `eos` tokens are added to the sentences.

- `lstm_dataset.py`
    Contains the code for creating the dataset for the LSTM model. The dataset is created using the `torch.utils.data.Dataset` class. It takes the sentences as contexts and the corresponding next word as targets accordingly. It also creates the vocabulary for the dataset using the training set.

- `lstm_model.py`
    Contains the code for the LSTM model. The model takes the input of the previous word and the hidden state of the previous time step and outputs the next word and the hidden state of the current time step. The model is trained using the `torch.nn` module. The activation function used is `ReLU` and the loss function used is `CrossEntropyLoss`.

- `lstm_train_test.py`
    Contains the code for training and testing the model. The model is trained for the optimal number of epochs using early stopping technique. It is then tested on the test set and the perplexity is calculated.

### Execution of Code

Make sure the following dependencies are installed:

- nltk
- numpy
- torch
- gensim

The code can be run by executing the following command:

```bash
python lstm_train_test.py
```

### Best Model

The best model after training is obtained by evaluation on the validation set. The model is stored in the `best_lstm_model.pth` file. This model can be restored by running the following command:

```python
model = mdl.LSTM_Model(embedding_matrix, vocab_size)
model.load_state_dict(torch.load('best_lstm_model.pth'))
```
