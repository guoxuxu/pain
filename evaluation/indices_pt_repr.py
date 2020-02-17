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
from gensim import corpora


BATCH_SIZE = 10
EPOCH = 20
LR = 0.001


class PainDataset(Dataset):
    def __init__(self, pos=True, transform=None):
        self.pos = pos
        self.transform = transform
        dataset = pickle.load(open('../../data/referral_dataset.pkl', 'rb'))
        documents = []
        pos_train_data = defaultdict(list)
        neg_train_data = defaultdict(list)
        labels = []
        for tuples in dataset.items():
            labels.append(tuples[0])
            for x, y in tuples[1]:
                sentence = [lemmatizer.lemmatize(word) for word in tokenizer.tokenize(x) if word not in stop_words]
                documents.append(sentence)
                if y == 'pos':
                    pos_train_data[tuples[0]].append(sentence)
                if y == 'neg':
                    neg_train_data[tuples[0]].append(sentence)
        dictionary = corpora.Dictionary(documents)
        if self.pos:
            data = []
            label = []
            for l in labels:
                for sentence in pos_train_data.get(l):
                    sen_indices = dictionary.doc2idx(sentence)
                    vector = np.pad(array(sen_indices), (0, 300-len(sen_indices)), 'constant')
                    vector = vector / 300.
                    vector = array(vector)[newaxis, :][newaxis, :]
                    data.append(vector)
                    label.append(float(l))
            self.data = array(data, dtype='f')
            self.label = array(label, dtype='f').reshape(len(label), 1)
        else:
            data = []
            label = []
            for l in labels:
                for sentence in neg_train_data.get(l):
                    sen_indices = dictionary.doc2idx(sentence)
                    vector = np.pad(array(sen_indices), (0, 300 - len(sen_indices)), 'constant')
                    vector = vector / 300.
                    vector = array(vector)[newaxis, :][newaxis, :]
                    data.append(vector)
                    label.append(float(l))
            self.data = array(data, dtype='f')
            self.label = array(label, dtype='f').reshape(len(label), 1)
        self.dictionary = dictionary

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
            nn.Linear(1*300, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
        )

        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 1*300),
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
            b_x = x.view(-1, 1*300)
            b_y = x.view(-1, 1*300)
            encoded, decoded = autoencoder(b_x)

            loss = loss_function(decoded, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 100 == 0:
                print('epoch:', epoch, '| train loss: ', loss.data.numpy())

    torch.save(autoencoder.state_dict(), '../../storeParams/autoencoder_indice_state.pt')


if __name__ == '__main__':
    train_encoder()
