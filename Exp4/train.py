import numpy as np

import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn

import torch

import os

from network import *


class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, real, atte, ideal_atte):
        return torch.mean(torch.pow((pred - real), 2))


class TestLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        return torch.mean(torch.pow((pred - real), 2))


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


custom_loss = CustomLoss()
testLoss = TestLoss()

BATCH_SIZE = 7000
learning_rate = 1e-3
epochs = 50

mini_loss = 1

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 1

# 模型有CNN_CNN，CNN_FCN，CNN_CNNFCN，CNN_RNN，CNN_LSTM
selected_model = 'CNN_LSTM'
model = CNN_LSTM()

noise_eeg = np.load('../data/train_input.npy')
clean_eeg = np.load('../data/train_output.npy')

noise1 = np.load('../data/EOG_all_epochs.npy')
noise2 = np.load('../data/EMG_all_epochs.npy')

noise1 = standardization(noise1)
noise2 = standardization(noise2)

noise = np.concatenate((noise1, noise2), axis=0)

indicator1 = np.zeros(noise_eeg.shape[0])
indicator2 = np.ones(noise.shape[0])
indicator3 = np.zeros(clean_eeg.shape[0])

indicator = np.concatenate((indicator1, indicator2, indicator3), axis=0)

train_input = np.concatenate((noise_eeg, noise, clean_eeg), axis=0)
train_output = np.concatenate((clean_eeg, noise, clean_eeg), axis=0)


test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

indicator = torch.from_numpy(indicator)
indicator = indicator.unsqueeze(1)

train_input = torch.from_numpy(train_input)
train_output = torch.from_numpy(train_output)


'''
注意使用不同的模型，embedding vector和attenuation vector的维度会有不同，要相应调整ideal_atte_x的长度
'''

ideal_atte_x_comp = np.array([0, 1])
ideal_atte_x = np.tile(ideal_atte_x_comp, 256)
ideal_atte_x = torch.from_numpy(ideal_atte_x)
ideal_atte_x = ideal_atte_x.float()

train_torch_dataset = Data.TensorDataset(train_input, indicator, train_output)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)

test_indicator = np.zeros(test_input.shape[0])
test_indicator = torch.from_numpy(test_indicator)
test_indicator = test_indicator.unsqueeze(1)

test_torch_dataset = Data.TensorDataset(test_input, test_indicator, test_output)

test_loader = Data.DataLoader(
    dataset=test_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,               # test set不要打乱数据
)

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model.to(device)  # 移动模型到cuda

optimizer = optim.Adam(model.parameters(), lr=learning_rate)


if os.path.exists('checkpoint/' + selected_model + '.pkl'):
    print('load model')
    model.load_state_dict(torch.load('checkpoint/' + selected_model + '.pkl'))

train_loss_list = []
test_loss_list = []
for epoch in range(epochs):

    train_acc = 0
    train_loss = 0

    total_train_loss_per_epoch = 0
    average_train_loss_per_epoch = 0
    train_step_num = 0

    for step, (train_input, indicator, train_output) in enumerate(train_loader):

        train_step_num += 1

        ideal_atte_x = ideal_atte_x.float().to(device)
        indicator = indicator.float().to(device)

        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)

        optimizer.zero_grad()

        train_preds, train_atte_x = model(train_input, indicator)

        train_loss = custom_loss(train_preds, train_output, train_atte_x, ideal_atte_x)

        total_train_loss_per_epoch += train_loss.item()

        train_loss.backward()
        optimizer.step()

    average_train_loss_per_epoch = total_train_loss_per_epoch / train_step_num

    if epoch % print_loss_frequency == 0:
        print('train loss: ', average_train_loss_per_epoch)

    test_step_num = 0
    total_test_loss_per_epoch = 0
    average_test_loss_per_epoch = 0

    if epoch % test_frequency == 0:

        for step, (test_input, test_indicator, test_output) in enumerate(test_loader):

            test_step_num += 1

            test_indicator = test_indicator.float().to(device)

            test_input = test_input.float().to(device)
            test_output = test_output.float().to(device)

            test_preds, test_atte_x = model(test_input, test_indicator)

            test_loss = testLoss(test_preds, test_output)

            total_test_loss_per_epoch += test_loss.item()

        average_test_loss_per_epoch = total_test_loss_per_epoch / test_step_num
        test_loss_list.append(average_test_loss_per_epoch)
        print('--------------test loss: ', average_test_loss_per_epoch)

        if average_test_loss_per_epoch < mini_loss:
            print('save model')
            torch.save(model.state_dict(), 'checkpoint/' + selected_model + '.pkl')
            mini_loss = average_test_loss_per_epoch

np.save("CNN-LSTM train loss",train_loss_list)
np.save("CNN-LSTM test loss",test_loss_list)