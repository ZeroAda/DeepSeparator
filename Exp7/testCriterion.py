import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import os
from network import *
from scipy.fftpack import fft


class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        return torch.mean(torch.pow((pred - real), 2))


def FourierLoss(pred, real):
    pred = pred.cpu()
    real = real.cpu()

    pred = pred.detach().numpy()
    real = real.detach().numpy()

    fft_pred = pred
    fft_real = real

    for i in range(len(pred)):
        fft_pred[i] = fft(pred[i])
        fft_real[i] = fft(real[i])

    abs_fft_pred = np.abs(fft_pred)
    abs_fft_real = np.abs(fft_real)

    return np.mean(np.square(abs_fft_pred - abs_fft_real))


def correlation(pred, real):
    pred = pred.cpu()
    real = real.cpu()

    pred = pred.detach().numpy()
    real = real.detach().numpy()
    
    result = 0
    
    for i in range(pred.shape[0]):
        result += np.corrcoef(pred[i], real[i])[0][1]

    final_result = result / pred.shape[0]

    return final_result


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


custom_loss = CustomLoss()

BATCH_SIZE = 1000
learning_rate = 1e-3
epochs = 1

mini_loss = 100

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 10

# 模型有CNN，FCN，RNN，LSTM
selected_model = 'LSTM'
model = LSTM()

# test_input = np.load('../data/test_input.npy')
# test_output = np.load('../data/test_output.npy')
# #
# test_input = np.load('../data/EOG_EEG_test_input.npy')
# test_output = np.load('../data/EOG_EEG_test_output.npy')
# # #
test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

'''
注意使用不同的模型，embedding vector和attenuation vector的维度会有不同，要相应调整ideal_atte_x的长度
'''

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

test_step_num = 0

total_test_loss_per_epoch = 0
average_test_loss_per_epoch = 0

total_test_fourier_loss_per_epoch = 0
average_test_fourier_loss_per_epoch = 0

total_test_correlation_per_epoch = 0
average_test_correlation_per_epoch = 0

for step, (test_input, test_output) in enumerate(test_loader):

    test_step_num += 1

    test_input = test_input.float().to(device)
    test_output = test_output.float().to(device)

    test_preds = model(test_input)

    test_loss = custom_loss(test_preds, test_output)
    test_fourier_loss = FourierLoss(test_preds, test_output)
    corr = correlation(test_preds, test_output)

    total_test_loss_per_epoch += test_loss.item()
    total_test_fourier_loss_per_epoch += test_fourier_loss

average_test_loss_per_epoch = total_test_loss_per_epoch / test_step_num
average_test_fourier_loss_per_epoch = total_test_fourier_loss_per_epoch / test_step_num

print('--------------test temporal MSE: ', average_test_loss_per_epoch)
print('--------------test spectral MSE: ', average_test_fourier_loss_per_epoch)
print('--------------test correlation: ', corr)

