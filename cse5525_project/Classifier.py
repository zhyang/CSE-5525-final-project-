import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embedding_weights, bidirectional = False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional = bidirectional)
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, label_size)
        else:
          self.fc = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.3)


    def forward(self, sentence, src_len, train = True):
        embeds = self.word_embeddings(sentence)
        # output, (hidden,cell) = self.lstm(embeds)
        # output = output.squeeze(0)
        # output = output[-1]
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)
        packed_outputs, (hidden,cell) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        output = hidden.squeeze(0)
        output = self.dropout(output)
        y = self.fc(output)
        return y

class RNNClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embedding_weights, bidirectional = False):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, bidirectional = bidirectional)
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, label_size)
        else:
          self.fc = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.3)


    def forward(self, sentence, src_len, train = True):
        embeds = self.word_embeddings(sentence)
        # rnn_out, hidden = self.rnn(embeds)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)
        packed_outputs, hidden = self.rnn(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        output = hidden.squeeze(0)
        output = self.dropout(output)
        y = self.fc(output)
        return y


class GRUClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, label_size, batch_size, embedding_weights, bidirectional = False):
        super(GRUClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding.from_pretrained(embedding_weights)
        self.gru = nn.GRU(embedding_dim, hidden_dim, bidirectional = bidirectional)
        if bidirectional:
          self.fc = nn.Linear(hidden_dim*2, label_size)
        else:
          self.fc = nn.Linear(hidden_dim, label_size)
        self.dropout = nn.Dropout(0.3)


    def forward(self, sentence, src_len, train = True):
        embeds = self.word_embeddings(sentence)
        # rnn_out, hidden = self.gru(embeds)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embeds, src_len)
        packed_outputs, hidden = self.gru(packed_embedded)
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        output = hidden.squeeze(0)
        output = self.dropout(output)
        y = self.fc(output)
        return y



class CNNClassifier(nn.Module):
    # def __init__(self, X, Y, VOCAB_SIZE, DIM_EMB=30, NUM_FILTERS=100, NUM_CLASSES=2):
    def __init__(self, DIM_EMB, NUM_CLASSES, NUM_FILTERS, embedding_weights):
        super(CNNClassifier, self).__init__()
        (self.DIM_EMB, self.NUM_CLASSES, self.NUM_FILTERS) = (DIM_EMB, NUM_CLASSES, NUM_FILTERS)
        #TODO: Initialize parameters.
        self.E = nn.Embedding.from_pretrained(embedding_weights)
        self.WINDOW_SIZES = [1,2,3]
        self.R = nn.ReLU()
        self.conv1 = nn.Conv1d(DIM_EMB, NUM_FILTERS, kernel_size=1)
        self.conv2 = nn.Conv1d(DIM_EMB, NUM_FILTERS, kernel_size=2)
        self.conv3 = nn.Conv1d(DIM_EMB, NUM_FILTERS, kernel_size=3)
        self.dropout = nn.Dropout(0.3)

        self.FC = nn.Linear(in_features=NUM_FILTERS*len(self.WINDOW_SIZES),
                            out_features=NUM_CLASSES)
        # self.FC = nn.Linear(in_features=NUM_FILTERS,
                            # out_features=NUM_CLASSES)

        self.softmax = nn.Softmax(dim=1)
        nn.init.xavier_uniform_(self.E.weight)
        nn.init.xavier_uniform_(self.FC.weight)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)


    def forward(self, sentence, src_len, train=True):
        #TODO: Implement forward computation.
        # X = X.unsqueeze(0)
        sentence = sentence.t()
        Sent_length = sentence.shape[1]
        M1 = nn.MaxPool1d(kernel_size=Sent_length-1+1)
        M2 = nn.MaxPool1d(kernel_size=Sent_length-2+1)
        M3 = nn.MaxPool1d(kernel_size=Sent_length-3+1)
        embedding = self.E(sentence)
        embedding = embedding.permute(0, 2, 1)
        X1 = self.R(M1(self.conv1(embedding)))
        X2 = self.R(M2(self.conv2(embedding)))
        X3 = self.R(M3(self.conv3(embedding)))
        
        output = torch.cat([X1, X2, X3], dim=1)
        # output = X3
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.FC(output)

        return self.softmax(output)
