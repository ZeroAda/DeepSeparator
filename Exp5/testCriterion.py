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

BATCH_SIZE = 100
learning_rate = 1e-3
epochs = 1

mini_loss = 100

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 10

sliding_window_length = 2048

data_amount = 8000
new_data_length = 2048
original_data_length = 512

# 模型有CNN_CNN，CNN_FCN，CNN_CNNFCN，CNN_RNN，CNN_LSTM
selected_model = 'CNN_CNN'
model = CNN_CNN()

test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')

# test_input = np.load('../data/EOG_EEG_test_input.npy')
# test_output = np.load('../data/EOG_EEG_test_output.npy')
#
# test_input = np.load('../data/EMG_EEG_test_input.npy')
# test_output = np.load('../data/EMG_EEG_test_output.npy')

final_input = np.zeros([data_amount, new_data_length])
final_output = np.zeros([data_amount, new_data_length])

for i in range(data_amount):
    x = np.linspace(0, original_data_length, original_data_length)
    y = test_input[i]

    # 这里是为了对y进行插值
    xvals = np.linspace(0, original_data_length, new_data_length)
    yinterp = np.interp(xvals, x, y)
    final_input[i] = yinterp

for i in range(data_amount):
    x = np.linspace(0, original_data_length, original_data_length)
    y = test_output[i]

    xvals = np.linspace(0, original_data_length, new_data_length)
    yinterp = np.interp(xvals, x, y)
    final_output[i] = yinterp

test_input = torch.from_numpy(final_input)
test_output = torch.from_numpy(final_output)

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


'''
选择加载不同的模型,有：FCN, CNNFCN, RNN, LSTM
注意加载不同的模型，保存的模型文件命名也要不一样 
'''


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


for step, (test_input, test_indicator, test_output) in enumerate(test_loader):

    for i in range(new_data_length // sliding_window_length):

        test_step_num += 1

        test_indicator = test_indicator.float().to(device)

        test_input = test_input.float().to(device)
        test_output = test_output.float().to(device)

        test_preds, _ = model(test_input[:, i*sliding_window_length:i*sliding_window_length+sliding_window_length],
                              test_indicator)

        test_loss = custom_loss(test_preds, test_output[:, i*sliding_window_length:i*sliding_window_length+sliding_window_length])
        test_fourier_loss = FourierLoss(test_preds, test_output[:, i*sliding_window_length:i*sliding_window_length+sliding_window_length])

        corr = correlation(test_preds, test_output[:, i*sliding_window_length:i*sliding_window_length+sliding_window_length])

        total_test_loss_per_epoch += test_loss.item()
        total_test_fourier_loss_per_epoch += test_fourier_loss

average_test_loss_per_epoch = total_test_loss_per_epoch / test_step_num
average_test_fourier_loss_per_epoch = total_test_fourier_loss_per_epoch / test_step_num

print('--------------test temporal MSE: ', average_test_loss_per_epoch)
print('--------------test spectral MSE: ', average_test_fourier_loss_per_epoch)
print('--------------test correlation: ', corr)

