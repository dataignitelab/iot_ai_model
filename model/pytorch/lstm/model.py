import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.num_layers = 2
        self.hidden_size = 64

        # in_channels, out_channels, kernel_size
        self.conv = nn.Conv1d(3, 16, 3, padding=1)
        self.maxpooling = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        self.maxpooling2 = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(32, self.hidden_size, num_layers=self.num_layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(64, 5, bias = True) 

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 예측을 위한 함수
    def forward(self, x):
        # print(x.shape)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.maxpooling(x)
        x = self.conv2(x)
        x = self.maxpooling2(x)
        x = x.transpose(1, 2)
        
        # h0 = torch.zeros(2, 500, self.hidden_size).to(self.device) # (BATCH SIZE, SEQ_LENGTH, HIDDEN_SIZE)
        # c0 = torch.zeros(2, 500, self.hidden_size).to(self.device) # hidden state와 동일
        
        x, _status = self.lstm(x) #, (h0, c0)
        x = self.fc(x[:, -1]) 
        return x