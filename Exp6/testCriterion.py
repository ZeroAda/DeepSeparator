'''
Log:
update RRMSE TEMPORAL SPECTRAL
'''
import numpy as np
import torch.optim as optim
import torch.utils.data as Data
import os
from network import *
from scipy.fftpack import fft
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import math

class CustomLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, pred, real):
        coe = get_rms(pred[0]-real[0])/get_rms(real[0])
        return coe


def FourierLoss(pred, real):
    pred = pred.cpu()
    real = real.cpu()

    pred = pred.detach().numpy()
    real = real.detach().numpy()
    print(type(pred))

    fft_pred = pred
    fft_real = real

    for i in range(len(pred)):
        fft_pred[i] = fft(pred[i])
        fft_real[i] = fft(real[i])

    abs_fft_pred = np.abs(fft_pred)
    abs_fft_real = np.abs(fft_real)

    return np.mean(np.square(abs_fft_pred - abs_fft_real))


def get_rms(records):
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))


def SpectralLoss(pred, real):
    pred = pred.cpu()
    real = real.cpu()

    pred = pred.detach().numpy()
    real = real.detach().numpy()


    _psd_pred,freq = plt.psd(pred[0], NFFT=512, Fs=256, pad_to=1024,
                             scale_by_freq=True)
    psd_pred = _psd_pred[1:]
    _psd_real,freq = plt.psd(real[0], NFFT=512, Fs=256, pad_to=1024,
                              scale_by_freq=True)
    psd_real = _psd_real[1:]
    coe = get_rms(psd_real - psd_pred) / get_rms(psd_real)
    return coe



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

BATCH_SIZE = 0
learning_rate = 1e-3
epochs = 1

mini_loss = 100

print_loss_frequency = 1
print_train_accuracy_frequency = 1
test_frequency = 10

# 模型有CNN_CNN，CNN_FCN，CNN_CNNFCN，CNN_RNN，CNN_LSTM
selected_model = 'CNN_CNN'
model = CNN_CNN()

test_input = np.load('../data/test_input.npy')
test_output = np.load('../data/test_output.npy')
# #
# test_input = np.load('../data/EOG_EEG_test_input.npy')
# test_output = np.load('../data/EOG_EEG_test_output.npy')
# #
# test_input = np.load('../data/EMG_EEG_test_input.npy')
# test_output = np.load('../data/EMG_EEG_test_output.npy')

'''
注意使用不同的模型，embedding vector和attenuation vector的维度会有不同，要相应调整ideal_atte_x的长度
'''

ideal_atte_x_comp = np.array([0, 1])
ideal_atte_x = np.tile(ideal_atte_x_comp, 256)
ideal_atte_x = torch.from_numpy(ideal_atte_x)
ideal_atte_x = ideal_atte_x.float()

test_input = torch.from_numpy(test_input)
test_output = torch.from_numpy(test_output)

test_indicator = np.zeros(test_input.shape[0])
test_indicator = torch.from_numpy(test_indicator)
test_indicator = test_indicator.unsqueeze(1)
#
# test_torch_dataset = Data.TensorDataset(test_input, test_indicator, test_output)
#
# test_loader = Data.DataLoader(
#     dataset=test_torch_dataset,
#     batch_size=BATCH_SIZE,
#     shuffle=False,               # test set不要打乱数据
# )

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


# change range() to change sample
# 0 to 400: EOG
# 4000 to 4400: EMG
sample = 400
# MSE temporal matrix
mset_list = np.zeros((10, 1))
# MSE spectral matrix
mses_list = np.zeros((10, 1))
# Correlation coefficient matrix
cc_list = np.zeros((10, 1))
for i in range(0,400):
    print("------- sample ", i, "----------")
    for j in range(10):
        print("--------- SNR", j - 7, "-----------")

        _test_indicator = test_indicator[i+400*j].float().to(device)

        _test_input = test_input[i+400*j].float().to(device)
        _test_output = test_output[i+400*j].float().to(device)

        test_preds, _ = model(_test_input.unsqueeze(0), _test_indicator)

        test_loss = custom_loss(test_preds, _test_output.unsqueeze(0))
        test_fourier_loss = SpectralLoss(test_preds, _test_output.unsqueeze(0))

        corr = correlation(test_preds, _test_output.unsqueeze(0))

        mset_list[j, 0] += test_loss
        mses_list[j, 0] += test_fourier_loss
        cc_list[j, 0] += corr

mset_list /= sample
mses_list /= sample
cc_list /= sample

np.savetxt("EOGmset matrix_CNN_test", mset_list)
np.savetxt("EOGmses matrix_CNN_test", mses_list)
np.savetxt("EOGcc matrix_CNN_test", cc_list)

