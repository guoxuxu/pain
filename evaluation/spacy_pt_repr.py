from collections import defaultdict
import pickle
import numpy as np
from numpy import array, newaxis
from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import spacy
nlp = spacy.load('en_core_web_lg')
from textblob.np_extractors import ConllExtractor
extractor = ConllExtractor()
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


## 1 :  pos num 151 ;neg num 223
# 2 :  pos num 9 ;neg num 365
## 3 :  pos num 60 ;neg num 314
# 4 :  pos num 21 ;neg num 353
## 5 :  pos num 324 ;neg num 50
## 6 :  pos num 69 ;neg num 305
# 7 :  pos num 23 ;neg num 351
# 8 :  pos num 6 ;neg num 368


EPOCH = 10
BATCH_SIZE = 10
LR = 0.001


class PainDataset(Dataset):

    def __init__(self, pos=True, transform=None):
        self.pos = pos
        datasets = pickle.load(open('../../data/referral_dataset.pkl', 'rb'))
        pos_train_data = defaultdict()
        neg_train_data = defaultdict()
        labels = []
        for tuples in datasets.items():
            labels.append(tuples[0])
            poslist = []
            neglist = []
            for x, y in tuples[1]:
                sentence = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x) if word not in stop_words]
                vectors = np.zeros(shape=(300, 300), dtype='f')
                for i in range(len(sentence)):
                    vectors[:, i] = nlp.vocab[sentence[i]].vector
                vectors = array(vectors)[newaxis, :, :]
                if y == 'pos':
                    poslist.append(vectors)
                if y == 'neg':
                    neglist.append(vectors)
            pos_train_data[tuples[0]] = poslist
            neg_train_data[tuples[0]] = neglist
        if self.pos:
            data = array(pos_train_data.get(labels[0]))
            label = np.ones(shape=(data.shape[0], labels[0]))
            for i in range(0, len(labels)):
                if i < len(labels) - 1:
                    next_data_batch = array(pos_train_data.get(labels[i+1]))
                    data = np.concatenate((data,next_data_batch),axis=0)
                    label = np.concatenate((label,np.ones(shape=(next_data_batch.shape[0],1))*labels[i]),
                                           axis=0)
            self.data = array(data)
            self.label = label
        else:
            data = array(neg_train_data.get(1))
            label = np.ones(shape=(data.shape[0], 1))
            for i in range(0, len(labels)):
                if i < len(labels) - 1:
                    next_data_batch = array(neg_train_data.get(labels[i+1]))
                    data = np.concatenate((data, next_data_batch), axis=0)
                    label = np.concatenate((label, np.ones(shape=(next_data_batch.shape[0], 1))*labels[i]),
                                           axis=0)
            self.data = array(data)
            self.label = label
        self.transform = transform

    @property
    def train_labels(self):
        return self.label

    @property
    def train_data(self):
        return self.data

    def __getitem__(self, index):

        if self.transform is not None:
            feature = self.transform(self.data)
            label = self.transform(self.label)
            feature, label = self.data[index], self.label[index]
            return feature, label
        else:
            feature, label = self.data[index], self.label[index]
            return feature, label

    def __len__(self):
        return len(self.data)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(300*300, 1024),
            nn.Tanh(),
            nn.Linear(1024, 512),
            nn.Tanh(),
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 512),
            nn.Tanh(),
            nn.Linear(512, 1024),
            nn.Tanh(),
            nn.Linear(1024, 300*300),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_encoder():
    pos_dataset = PainDataset(pos=True)
    pos_loader = DataLoader(dataset=pos_dataset, batch_size=BATCH_SIZE, shuffle=True)

    autoencoder = AutoEncoder()

    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    loss_function = nn.MSELoss()

    for epoch in range(EPOCH):
        for step, (x, x_label) in enumerate(pos_loader):
            b_x = x.view(-1, 300 * 300)
            b_y = x.view(-1, 300 * 300)
            encoded, decoded = autoencoder(b_x)

            loss = loss_function(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('epoch:', epoch, '| train loss: ', loss.data.numpy())

    torch.save(autoencoder.state_dict(), '../../storeParams/autoencoder_state.pt')


if __name__ == '__main__':
    train_encoder()
