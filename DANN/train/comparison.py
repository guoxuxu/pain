import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn import metrics
from collections import defaultdict
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 64
LR = 0.005         # learning rate
DOWNLOAD_MNIST = False
N_TEST_IMG = 5
ENCODE = 8

# Mnist digits dataset
train_data = torchvision.datasets.MNIST(
    root='../data/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),    # Converts a PIL.Image or numpy.ndarray to
    download=DOWNLOAD_MNIST,                        # download it if you don't have it
)

test_data = torchvision.datasets.MNIST(
    root='../data/',
    train=False,  # this is training data
    transform=torchvision.transforms.ToTensor(),  # Converts a PIL.Image or numpy.ndarray to
    download=DOWNLOAD_MNIST,  # download it if you don't have it
)

train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = Data.DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=True)

# classifiers
names = ["Nearest_Neighbors", "Linear_SVM", "RBF_SVM",
             "Decision_Tree", "Random_Forest", "Neural_Net", "AdaBoost",
             "Naive_Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

save_acc = defaultdict(list)


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, ENCODE),   #
        )
        self.decoder = nn.Sequential(
            nn.Linear(ENCODE, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class DAutoEncoder(nn.Module):
    def __init__(self):
        super(DAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 12),
            nn.Tanh(),
            nn.Linear(12, ENCODE),   #
        )
        self.decoder = nn.Sequential(
            nn.Linear(ENCODE, 12),
            nn.Tanh(),
            nn.Linear(12, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 28*28),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
        # self.branch = nn.Linear(ENCODE, 2)
        self.branch = nn.Sequential(
            nn.Linear(ENCODE, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        classifier = self.branch(encoded)
        return classifier, encoded, decoded

# ANN
Basic_AE = AutoEncoder().cuda()
optimizer_AE = torch.optim.Adam(Basic_AE.parameters(), lr=LR)

# DANN
autoencoder = DAutoEncoder().cuda()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)

loss_func_1 = nn.MSELoss()
# loss_func_2 = nn.CrossEntropyLoss()
loss_func_2 = nn.MSELoss()

def train_ae():
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28*28).cuda()   # batch x, shape (batch, 28*28)
            encoded, decoded = Basic_AE(b_x)
            loss = loss_func_1(decoded, b_x)      # mean square error
            optimizer_AE.zero_grad()               # clear gradients for this training step
            loss.backward()                     # backpropagation, compute gradients
            optimizer_AE.step()                    # apply gradients

            if step % 100 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.detach().cpu().numpy().mean())

    torch.save(Basic_AE.state_dict(), '../models/ANN.pt')

def train_dae(gamma = 2):
    for epoch in range(EPOCH):
        for step, (x, b_label) in enumerate(train_loader):
            b_x = x.view(-1, 28*28).cuda()   # batch x, shape (batch, 28*28)
            b_y = (b_label % 2).float().cuda()
            classifier, _,  decode = autoencoder(b_x)
            l1 = loss_func_1(decode, b_x)
            l2 = loss_func_2(classifier, b_y)
            total_loss = l1 + gamma * l2
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print('Epoch: ', epoch, '| AE loss: %.4f' % l1.detach().cpu().item(), '| D loss: %.4f' %l2.detach().cpu().item())

    torch.save(autoencoder.state_dict(), '../models/DANN.pt')

def test():
    # test
    X = test_data.data
    y = test_data.targets

    D_model = DAutoEncoder()
    D_model.load_state_dict(torch.load('../models/DANN.pt'))
    D_model.eval()

    encode_feature = []
    for x in X:
        x_view = x.float().view(-1, 28 * 28)
        _, encode, _ = D_model(x_view)
        encode_feature.append(encode.reshape(ENCODE, ).data.numpy())
    encode_feature = np.array(encode_feature)
    encode_feature = encode_feature.reshape((len(encode_feature), -1))

    X_train, X_test, y_train, y_test = train_test_split(encode_feature, y.data.numpy(), test_size=0.3, random_state=1)

    print('\n')
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('(1):==DANN==============', name, score)

    ANN_model = AutoEncoder().cuda()
    ANN_model.load_state_dict(torch.load('../models/ANN.pt'))
    ANN_model.eval()

    encode_feature = []
    for x in X:
        x_view = x.float().view(-1, 28 * 28).cuda()
        encode, _ = ANN_model(x_view)
        encode_feature.append(encode.reshape(ENCODE, ).cpu().detach().numpy())
    encode_feature = np.array(encode_feature)
    encode_feature = encode_feature.reshape((len(encode_feature), -1))

    X_train, X_test, y_train, y_test = train_test_split(encode_feature, y.data.numpy(), test_size=0.3, random_state=1)

    print('\n')
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print('(2):==ANN==============', name, score)

    X_train, X_test, y_train, y_test = train_test_split(X.numpy().reshape((len(X),-1)), y.data.numpy(), test_size=0.3, random_state=1)
    
    pca = PCA(n_components=ENCODE)
    pca.fit_transform(X_train)
    pca_train = pca.transform(X_train)
    pca_test = pca.transform(X_test)
    
    print('\n')
    for name, clf in zip(names, classifiers):
        clf.fit(pca_train, y_train)
        score = clf.score(pca_test, y_test)
        print('(3):==PCA===============',name, score)



if __name__ == "__main__":
    #train_ae()
    # train_dae()
    test()

