import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from optimalNetwork import *

import seaborn as sns


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


BATCH_SIZE = 6000
learning_rate = 1e-5
epochs = 1

threshold = 0.5

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 1
save_model = 10

mini_loss = 100
maxauc = 0.92

show_test_detail = False
plot_loss = False

Loss_list = []
Accuracy_list = []

test_input = np.load('data/test_input.npy')
test_output = np.load('data/test_output.npy')

ideal_atte_x_comp = np.array([0, 1])
ideal_atte_x = np.tile(ideal_atte_x_comp, 1024)
ideal_atte_x = torch.from_numpy(ideal_atte_x)
ideal_atte_x = ideal_atte_x.float()

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)

test_indicator = np.ones(test_input.shape[0])
test_indicator = torch.from_numpy(test_indicator)
test_indicator = test_indicator.unsqueeze(1)

test_torch_dataset = Data.TensorDataset(test_input, test_indicator, test_output)

test_loader = Data.DataLoader(
    dataset=test_torch_dataset,      # torch TensorDataset format
    batch_size=BATCH_SIZE,      # mini batch size
    shuffle=False,               # 要不要打乱数据 (打乱比较好)
    # num_workers=2,              # 多线程来读数据
)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = CNN_CNN()
model.to(device)  # 移动模型到cuda


if os.path.exists('checkpoint/CNN_CNN_end2end.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/CNN_CNN_end2end.pkl'))


for step, (test_input, test_indicator, test_output) in enumerate(test_loader):

    ideal_atte_x = ideal_atte_x.float().to(device)
    test_indicator = test_indicator.float().to(device)

    test_input = test_input.float().to(device)
    test_output = test_output.float().to(device)

    test_preds0, atte_x0 = model(test_input, test_indicator, 0)
    test_preds1, atte_x1 = model(test_input, test_indicator, 0.1)
    test_preds2, atte_x2 = model(test_input, test_indicator, 1)
    test_preds3, atte_x3 = model(test_input, test_indicator, 10)
    test_preds4, atte_x4 = model(test_input, test_indicator, 100)

    test_preds0 = test_preds0.cpu()
    test_preds0 = test_preds0.detach().numpy()

    test_preds1 = test_preds1.cpu()
    test_preds1 = test_preds1.detach().numpy()

    test_preds2 = test_preds2.cpu()
    test_preds2 = test_preds2.detach().numpy()

    test_preds3 = test_preds3.cpu()
    test_preds3 = test_preds3.detach().numpy()

    test_preds4 = test_preds4.cpu()
    test_preds4 = test_preds4.detach().numpy()

    test_preds0 = test_preds0[100]
    test_preds1 = test_preds1[100]
    test_preds2 = test_preds2[100]
    test_preds3 = test_preds3[100]
    test_preds4 = test_preds4[100]

    l0, = plt.plot(test_preds0)
    l1, = plt.plot(test_preds1)
    l2, = plt.plot(test_preds2)
    l3, = plt.plot(test_preds3)
    l4, = plt.plot(test_preds4)

    plt.legend([l0, l1, l2, l3, l4], ['origin', '+0.1', '+1', '+10', '+100'], loc='upper right')

    #plt.show()
    plt.savefig('./noise+1.jpg')

