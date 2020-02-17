import torch
import numpy as np
from numpy import array
from collections import defaultdict
from torch.utils.data import DataLoader
from indices_pt_repr import AutoEncoder, PainDataset
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics


model = AutoEncoder()
model.load_state_dict(torch.load('../../storeParams/autoencoder_indice_state.pt'))
model.eval()

BATCH_SIZE = 10
pos_dataset = PainDataset(pos=True)
neg_dataset = PainDataset(pos=True)

acc_display = defaultdict(list)

for i in range(1, 9):
    un_pos_data = pos_dataset.train_data.reshape(663,300,)[pos_dataset.train_labels.transpose()[0]==i]
    un_neg_data = neg_dataset.train_data.reshape(663,300,)[neg_dataset.train_labels.transpose()[0]==i]
    pos_label = np.ones(shape=(len(un_pos_data),))
    neg_label = np.zeros(shape=(len(un_neg_data),))

    X_train, X_test, y_train, y_test = train_test_split(np.vstack((un_pos_data, un_neg_data)),
                                                        np.hstack((pos_label, neg_label)), test_size=0.3, random_state=1)

    gnb = GaussianNB()
    gnb.fit(X_train, y_train)
    y_pred = gnb.predict(X_test)

    acc_display[i].append(metrics.accuracy_score(y_test, y_pred))


print('-----------------------------------')

pos_loader = DataLoader(dataset=pos_dataset, batch_size=BATCH_SIZE, shuffle=True)
neg_loader = DataLoader(dataset=neg_dataset, batch_size=BATCH_SIZE, shuffle=True)

en_pos_data = []
pos_target = []
for step, (x, label) in enumerate(pos_loader):
    b_x = x.view(-1, 1*300)
    encoded, decoded = model(b_x)
    for i, vector in enumerate(encoded.detach().numpy()):
        en_pos_data.append(vector)
        pos_target.append(label[i].numpy())
en_pos_data = array(en_pos_data)
pos_target = array(pos_target)

print(en_pos_data.shape, pos_target.shape)

en_neg_data = []
neg_target = []
for step, (x, label) in enumerate(pos_loader):
    b_x = x.view(-1, 1*300)
    encoded, decoded = model(b_x)
    for i, vector in enumerate(encoded.detach().numpy()):
        en_neg_data.append(vector)
        neg_target.append(label[i].numpy())
en_neg_data = array(en_neg_data)
neg_target = array(neg_target)

print(en_neg_data.shape, neg_target.shape)

# combine all the dataset for ensemble binary classification

dataset = defaultdict(list)
target = defaultdict(list)
pos_target = pos_target.transpose()[0]
neg_target = neg_target.transpose()[0]
for i in range(0, len(pos_target)):

    for j in range(1,9):

        if pos_target[i] == j:
            dataset[j].append(en_pos_data[i])
            target[j].append(1)

        if neg_target[i] == j:
            dataset[j].append(en_neg_data[i])
            target[j].append(0)

for i in range(1, 9):
    data = array(dataset.get(i))
    if target.get(i):
        label = array(target.get(i)).reshape(len(target.get(i)), 1)
        X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=1)

        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)

        acc_display[i].append(metrics.accuracy_score(y_test, y_pred))


for key in acc_display:
    print('class', key, 'unencode accuracy', acc_display.get(key)[0], 'encode accuracy', acc_display.get(key)[1])
