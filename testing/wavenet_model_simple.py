import torch
import torch.nn as nn

class WaveNetModelSimple(nn.Module):

    def __init__(self, input_dim=48000, hidden_dim=600):
        super(WaveNetModelSimple, self).__init__()
        self.firstConv = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim , kernel_size=17, stride=1)
        self.endConv = nn.Conv2d(in_channels=hidden_dim, out_channels=50, kernel_size=(11, 7), stride=1)
        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax()

    def forward(self, input):
        b, l, w = input.shape

        #print(input.shape)
        input = input.float()
        out1 = self.relu(self.firstConv(input))
        #print(out1.shape)
        out1 = torch.reshape(out1, (b, 600, 15, 16))
        out2 = self.softmax(self.endConv(out1))
        #print(out2.shape)
        #return torch.transpose(torch.reshape(out2, (b, 300, 50)), 2, 1)
        return torch.reshape(out2, (b, 50, 50))