import numpy as np

import torch.optim as optim
import torch.utils.data as Data

import os

from network import *


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


loss = torch.nn.MSELoss(reduction='mean')

BATCH_SIZE = 2000
learning_rate = 1e-3
epochs = 1000

mini_loss = 100

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 10

# 模型有CNN，FCN，RNN，LSTM
selected_model = 'LSTM'
model = LSTM()

train_input = np.load('../data/train_input.npy')
train_output = np.load('../data/train_output.npy')

test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

train_input = torch.from_numpy(train_input)
train_output = torch.from_numpy(train_output)


train_torch_dataset = Data.TensorDataset(train_input, train_output)

train_loader = Data.DataLoader(
    dataset=train_torch_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)

test_torch_dataset = Data.TensorDataset(test_input, test_output)

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


for epoch in range(epochs):

    train_acc = 0
    train_loss = 0

    total_train_loss_per_epoch = 0
    average_train_loss_per_epoch = 0
    train_step_num = 0

    for step, (train_input, train_output) in enumerate(train_loader):

        train_step_num += 1

        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)

        optimizer.zero_grad()

        train_preds = model(train_input)

        train_loss = loss(train_preds, train_output)

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

        for step, (test_input, test_output) in enumerate(test_loader):

            test_step_num += 1

            test_input = test_input.float().to(device)
            test_output = test_output.float().to(device)

            test_preds = model(test_input)

            test_loss = loss(test_preds, test_output)

            total_test_loss_per_epoch += test_loss.item()

        average_test_loss_per_epoch = total_test_loss_per_epoch / test_step_num

        print('--------------test loss: ', average_test_loss_per_epoch)

        if average_test_loss_per_epoch < mini_loss:
            print('save model')
            torch.save(model.state_dict(), 'checkpoint/' + selected_model + '.pkl')
            mini_loss = average_test_loss_per_epoch
