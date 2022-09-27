import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self):
        super(LSTMModel, self).__init__()
        # self.hidden_dim = hidden_dim
        # self.seq_len = seq_len
        # self.output_dim = output_dim
        # self.layers = layers
        

        self.conv = nn.Conv1d(3, 16, 3, padding='same')
        self.maxpooling = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, 3, padding='same')
        self.maxpooling2 = nn.MaxPool1d(2)

        # self.conv3 = nn.Conv1d(16, 32, 3, padding='same')
        # self.maxpooling3 = nn.MaxPool1d(2)

        # torch.Size([3, 1, 250])
        

        self.lstm = nn.LSTM(32, 64, num_layers=2,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(64, 5, bias = True) 
        
    # 학습 초기화를 위한 함수
    # def reset_hidden_state(self): 
    #     self.hidden = (
    #             torch.zeros(self.layers, self.seq_len, self.hidden_dim),
    #             torch.zeros(self.layers, self.seq_len, self.hidden_dim))
    
    # 예측을 위한 함수
    def forward(self, x):
        # print(x.shape)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        # x = self.conv3(x)
        # x = self.maxpooling3(x)
        x = x.transpose(1, 2)
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x