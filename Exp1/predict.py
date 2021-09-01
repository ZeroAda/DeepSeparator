import numpy as np

import matplotlib.pyplot as plt

import torch.nn as nn
import os
import torch
import torch.utils.data as Data

from network import *


class My_loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, train_input, train_output):
        return torch.mean(torch.pow((train_input - train_output), 2))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


criterion = My_loss()

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

model = net()
model.to(device)  # 移动模型到cuda


if os.path.exists('checkpoint/model.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/model.pkl'))


for step, (test_input, test_indicator, test_output) in enumerate(test_loader):

    ideal_atte_x = ideal_atte_x.float().to(device)
    test_indicator = test_indicator.float().to(device)

    test_input = test_input.float().to(device)
    test_output = test_output.float().to(device)

    test_preds = model(test_input, test_indicator, ideal_atte_x)

    test_preds = test_preds.cpu()
    test_output = test_output.cpu()
    test_input = test_input.cpu()

    test_preds = test_preds.detach().numpy()
    test_output = test_output.detach().numpy()
    test_input = test_input.detach().numpy()

    test_input = test_input[60]
    test_preds = test_preds[60]
    test_output = test_output[60]

    l1, = plt.plot(test_input)
    l2, = plt.plot(test_preds)
    l3, = plt.plot(test_output)

    plt.legend([l1, l2, l3], ['noisy input', 'noise', 'ideal output'], loc='upper right')

    plt.title('denoise network')

    plt.show()


