import torch
import torch.nn as nn




class CNN_CNN(nn.Module):

    def __init__(self):
        super(CNN_CNN, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv1_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.conv1_6 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=1)

        self.conv2_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv2_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.conv2_6 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=1)

        self.conv3_1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.conv3_5 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, padding=2)
        self.conv3_6 = nn.Conv1d(in_channels=64, out_channels=1, kernel_size=5, padding=1)

        self.batch_norm = nn.BatchNorm1d(512, affine=True)

    def forward(self, x, indicator):

        emb_x = x
        learnable_atte_x = x

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

        emb_x = self.conv1_6(emb_x)

        emb_x = torch.squeeze(emb_x, 1)


        #########################

        learnable_atte_x = torch.unsqueeze(learnable_atte_x, 1)

        learnable_atte_x = self.conv3_1(learnable_atte_x)
        learnable_atte_x = torch.relu(learnable_atte_x)

        learnable_atte_x = self.conv3_2(learnable_atte_x)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)

        learnable_atte_x = self.conv3_3(learnable_atte_x)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)

        learnable_atte_x = self.conv3_4(learnable_atte_x)
        learnable_atte_x = torch.relu(learnable_atte_x)

        learnable_atte_x = self.conv3_5(learnable_atte_x)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)

        learnable_atte_x = self.conv3_6(learnable_atte_x)
        learnable_atte_x = torch.sigmoid(learnable_atte_x)

        learnable_atte_x = torch.squeeze(learnable_atte_x, 1)

        #########################

        atte_x = indicator - learnable_atte_x
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

        output = self.conv2_6(output)

        output = torch.squeeze(output, 1)

        return output, learnable_atte_x

