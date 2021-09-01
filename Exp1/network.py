import torch
import torch.nn as nn


class FCN(nn.Module):

    def __init__(self):
        super(FCN, self).__init__()

        self.fc1_1 = nn.Linear(512, 2048)
        self.fc1_2 = nn.Linear(2048, 2048)
        self.fc1_3 = nn.Linear(2048, 2048)
        self.fc1_4 = nn.Linear(2048, 2048)
        self.fc1_5 = nn.Linear(2048, 2048)
        self.fc1_6 = nn.Linear(2048, 2048)
        self.fc1_7 = nn.Linear(2048, 512)
        
        self.fc2_1 = nn.Linear(512, 2048)
        self.fc2_2 = nn.Linear(2048, 2048)
        self.fc2_3 = nn.Linear(2048, 2048)
        self.fc2_4 = nn.Linear(2048, 2048)
        self.fc2_5 = nn.Linear(2048, 2048)
        self.fc2_6 = nn.Linear(2048, 2048)
        self.fc2_7 = nn.Linear(2048, 512)

        self.batch_norm = nn.BatchNorm1d(2048, affine=True)

    def forward(self, x, indicator, ideal_atte_x):

        emb_x = x

        emb_x = self.fc1_1(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.fc1_2(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.fc1_3(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.batch_norm(emb_x)

        emb_x = self.fc1_4(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.fc1_5(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.batch_norm(emb_x)

        emb_x = self.fc1_6(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.fc1_7(emb_x)
        
        #########################

        atte_x = indicator - ideal_atte_x
        atte_x = torch.abs(atte_x)

        output = torch.mul(emb_x, atte_x)

        output = self.fc2_1(output)
        output = torch.relu(output)

        output = self.fc2_2(output)
        output = torch.relu(output)

        output = self.fc2_3(output)
        output = torch.sigmoid(output)
        
        output = self.batch_norm(output)

        output = self.fc2_4(output)
        output = torch.sigmoid(output)

        output = self.fc2_5(output)
        output = torch.relu(output)

        output = self.fc2_6(output)
        output = torch.sigmoid(output)

        output = self.fc2_7(output)

        return output


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.conv1_6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_7 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_9 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_10 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=1)

        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.conv2_6 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_7 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_8 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_9 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_10 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=1)

        self.batch_norm = nn.BatchNorm1d(512, affine=True)

    def forward(self, x, indicator, ideal_atte_x):

        emb_x = x

        emb_x = torch.unsqueeze(emb_x, 1)

        emb_x = self.conv1_1(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.conv1_2(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.conv1_3(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.conv1_4(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.conv1_5(emb_x)
        emb_x = torch.sigmoid(emb_x)

        #emb_x = self.conv1_6(emb_x)
        #emb_x = torch.relu(emb_x)

        #emb_x = self.conv1_7(emb_x)
        #emb_x = torch.sigmoid(emb_x)

        #emb_x = self.conv1_8(emb_x)
        #emb_x = torch.sigmoid(emb_x)

        #emb_x = self.conv1_9(emb_x)
        #emb_x = torch.sigmoid(emb_x)

        emb_x = self.conv1_10(emb_x)

        emb_x = torch.squeeze(emb_x, 1)

        #########################

        atte_x = indicator - ideal_atte_x
        atte_x = torch.abs(atte_x)

        output = torch.mul(emb_x, atte_x)

        #########################

        output = torch.unsqueeze(output, 1)

        output = self.conv2_1(output)
        output = torch.relu(output)

        output = self.conv2_2(output)
        output = torch.sigmoid(output)

        output = self.conv2_3(output)
        output = torch.sigmoid(output)

        output = self.conv2_4(output)
        output = torch.relu(output)

        output = self.conv2_5(output)
        output = torch.sigmoid(output)

        #output = self.conv2_6(output)
        #output = torch.relu(output)

        #output = self.conv2_7(output)
        #output = torch.sigmoid(output)

        #output = self.conv2_8(output)
        #output = torch.sigmoid(output)

        #output = self.conv2_9(output)
        #output = torch.sigmoid(output)

        output = self.conv2_10(output)

        output = torch.squeeze(output, 1)

        return output


class CNNFCN(nn.Module):

    def __init__(self):
        super(CNNFCN, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

        self.fc1_1 = nn.Linear(512, 512)
        self.fc1_2 = nn.Linear(512, 512)
        self.fc1_3 = nn.Linear(512, 512)
        self.fc1_4 = nn.Linear(512, 512)

        self.fc2_1 = nn.Linear(512, 512)
        self.fc2_2 = nn.Linear(512, 512)
        self.fc2_3 = nn.Linear(512, 512)
        self.fc2_4 = nn.Linear(512, 512)

        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, padding=1)

        self.batch_norm = nn.BatchNorm1d(512, affine=True)

    def forward(self, x, indicator, ideal_atte_x):

        emb_x = x
        emb_x = torch.unsqueeze(emb_x, 1)
        
        emb_x = self.conv1_1(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.conv1_2(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.conv1_3(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.conv1_4(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.conv1_5(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = torch.squeeze(emb_x, 1)
        
        emb_x = self.fc1_1(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.fc1_2(emb_x)
        emb_x = torch.sigmoid(emb_x)

        emb_x = self.fc1_3(emb_x)
        emb_x = torch.relu(emb_x)

        emb_x = self.fc1_4(emb_x)
        emb_x = torch.relu(emb_x)

        #########################

        atte_x = indicator - ideal_atte_x
        atte_x = torch.abs(atte_x)

        output = torch.mul(emb_x, atte_x)

        output = self.fc2_1(output)
        output = torch.relu(output)

        output = self.fc2_2(output)
        output = torch.relu(output)

        output = self.fc2_3(output)
        output = torch.sigmoid(output)

        output = self.fc2_4(output)
        output = torch.sigmoid(output)

        output = torch.unsqueeze(output, 1)

        output = self.conv2_1(output)
        output = torch.relu(output)

        output = self.conv2_2(output)
        output = torch.sigmoid(output)

        output = self.conv2_3(output)
        output = torch.sigmoid(output)

        output = self.conv2_4(output)
        output = torch.sigmoid(output)

        output = self.conv2_5(output)

        output = torch.squeeze(output, 1)

        return output


class RNN(nn.Module):

    def __init__(self):
        super(RNN, self).__init__()

        self.rnn1 = nn.RNN(input_size=1, hidden_size=1, num_layers=1, dropout=0.5)
        self.rnn2 = nn.RNN(input_size=1, hidden_size=1, num_layers=1, dropout=0.5)

    def forward(self, x, indicator, ideal_atte_x):
        
        x = x.t()
        x = torch.unsqueeze(x, 2)
        
        #print('emb_x before RNN: ', x.shape)
        
        emb_x = x
        emb_x, hidden = self.rnn1(emb_x, None)

        #print('emb_x after RNN: ', emb_x.shape)
        #########################

        atte_x = indicator - ideal_atte_x
        atte_x = torch.abs(atte_x)

        emb_x = torch.squeeze(emb_x, 2)
        emb_x = emb_x.t()

        output = torch.mul(emb_x, atte_x)

        output = output.t()
        output = torch.unsqueeze(output, -1)

        output, hidden = self.rnn2(output, None)

        output = torch.squeeze(output, -1)
        output = output.t()

        return output


class LSTM(nn.Module):

    def __init__(self):
        super(LSTM, self).__init__()

        self.lstm1 = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=1, hidden_size=1, num_layers=1, dropout=0.5)

    def forward(self, x, indicator, ideal_atte_x):

        x = x.t()
        x = torch.unsqueeze(x, -1)

        emb_x = x
        emb_x, hidden = self.lstm1(emb_x, None)
        
        emb_x = torch.squeeze(emb_x, -1)
        emb_x = emb_x.t()

        #########################

        atte_x = indicator - ideal_atte_x
        atte_x = torch.abs(atte_x)

        output = torch.mul(emb_x, atte_x)
        
        output = output.t()
        output = torch.unsqueeze(output, -1)
        
        output, hidden = self.lstm2(output, None)
        
        output = torch.squeeze(output, -1)
        output = output.t()
        
        return output
