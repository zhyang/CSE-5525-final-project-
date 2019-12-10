import torch
import random
import numpy as np
from datareader import DataReader
import torch.optim as optim
import torch.nn as nn
from sklearn import metrics
import Classifier as C

data_path = 'data1/'
Train_iter = 30
batch_size = 16
nlabel = 2
hidden_dim = 25
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random_seed = 1234
BiDirection = True
nfilter = 100
use_gpu = torch.cuda.is_available()
embedding = 'glove.6B.300d'
random.seed(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
torch.backends.cudnn.deterministic = True

use_rnn = False

def Train(data, epochs, batch_size):

    if use_rnn:
        model = C.GRUClassifier(embedding_dim=data.embedding.shape[1],hidden_dim=hidden_dim,
                           label_size=nlabel, batch_size=batch_size, embedding_weights=data.embedding,  bidirectional = BiDirection)
    else:
        model = C.CNNClassifier(DIM_EMB=data.embedding.shape[1], NUM_CLASSES=nlabel, NUM_FILTERS=nfilter, 
                                embedding_weights=data.embedding)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        total_loss = 0.0
        for i, batch in enumerate(data.train_iter):
            # feature, label = batch.sentence, batch.label
            (feature, batch_length), label = batch.sentence, batch.label
            optimizer.zero_grad()
            output = model(feature, batch_length)
            # print(output)
            loss = loss_function(output, label)
            total_loss += loss
            loss.backward()
            optimizer.step()
        print(f"loss on epoch {epoch} = {total_loss}")
    return model


def Eval(data, MLP):
    num_correct = 0
    total = 0
    LL = []
    PL = []
    for i, batch in enumerate(data.dev_b_iter):
        (feature, batch_length), label = batch.sentence, batch.label
        Probs = MLP.forward(feature, batch_length, train=False)
        Probs = Probs.view(label.shape[0],-1)
        # print(Probs.shape)
        _, prediction = torch.max(Probs, 1)
        LL = LL + label.tolist()
        PL = PL + prediction.tolist()
        num_correct += (prediction == label).sum()
        total += len(label)
    # F1Score = metrics.f1_score(LL, PL, average='weighted')  
    # print(LL)
    # print(PL)
    F1Score = metrics.f1_score(LL, PL)  
    print("Accuracy: %s" % (float(num_correct) / float(total)))
    print('F1 score: %s' % F1Score)



if __name__ == "__main__":
    data = DataReader(data_path=data_path, batch_size=batch_size, device=device, embedding=embedding)
    MLP = Train(data, Train_iter, batch_size)
    Eval(data, MLP)

