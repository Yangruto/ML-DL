import numpy as np
import pandas as pd
from typing import Union
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

feature_size = 12 # time step
input_size = 1 # feature size
batch_size = 12
hidden_size = 512
num_layers = 1

learning_rate = 0.0001
training_step = 2000

training_feature = []
target_col = ''

def normalize_data(data:pd.DataFrame, col:str) -> pd.DataFrame:
    """
    Normalize data by MinMaxScaler.
        data: pandas dataframe
        col: normalize column
    """
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(data[[col]])
    data[col] = scaler.transform(data[[col]])
    return scaler, data

def lag_data(data:pd.DataFrame, col:str, lag_n:int) -> pd.DataFrame:
    """
    Prepare lag data.
        data: dataframe
        col: lag column
        lag_n: lag number
    """
    for i in range(1, lag_n + 1):
        data[f'{col}_lag_{i}'] = data[col].shift(i)
    data = data.loc[data[f'{col}_lag_{i}'].notnull()]
    data = data.reset_index(drop=True)
    return data

class LSTM_MODEL(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_layers:int) -> nn.Module:
        """
        Create a pytorch LSTM model frame.
            input_size: feature size
            hidden_size: lstm hidden layer parameter size
            num_layers: lstm hidden layer size
        """
        super(LSTM_MODEL, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.feature_size = feature_size
        self.batch_size = batch_size

        self.lstm_1 = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.lstm_2 = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.dropout_1 = nn.Dropout(0.2)
        self.fc_1 = nn.Linear(self.hidden_size * self.feature_size, self.hidden_size)
        self.fc_2 = nn.Linear(self.hidden_size, 1)
        self.relu = nn.ReLU()
        self.norm_1 = nn.LayerNorm([self.feature_size, self.input_size]) # the dimensions must include all except batch size 
        self.norm_2 = nn.LayerNorm([self.hidden_size])
        
    def forward(self, x:torch.tensor) -> torch.tensor:
        """
        Calculate forward propagation.
            x: data for LSTM
        """
        # x shape: (batch, feature_size, input_size), (12, 12, 1)
        output = self.norm_1(x)
        # norm output shape: (batch, feature_size, input_size) (12, 12, 1)
        output, (hn_0, cn_0) = self.lstm_1(output)
        # lstm1 output shape: (batch, feature_size, hiddent_size)  (12, 12, 512)
        output, (hn_1, cn_1) = self.lstm_2(output)
        # lstm2 output shape: (batch, feature_size, hiddent_size)  (12, 12, 512)
        # reshaping the data for Dense layer
        output = output.reshape(len(output), -1)
        output = self.fc_1(output) 
        output = self.norm_2(output)
        output = self.relu(output)
        # output = self.dropout_1(output)
        output = self.fc_2(output) 
        return output

    def define_optimizer(self, learning_rate:float) -> None:
        """
        Define optimizer and learning rate for LSTM model.
            lr: learning rate
        """
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.parameters(), lr=self.learning_rate)
    
    def define_loss_function(self) -> None:
        self.loss_function = nn.MSELoss()

    def training_(self, step_time:int) -> None:
        """
        Train LSTM model.
            step_time: training step time
        """
        for step in range(1, step_time + 1):
            for data, target in self.training_data:
                self.optimizer.zero_grad()
                predict = self.forward(data)
                loss = self.loss_function(predict, target)
                loss.backward()
                self.optimizer.step()
            if step % 10 == 0:
                print(f'Step:{step} Loss:{loss:.5E}') 
            # update learning
            # if (past_loss <= loss) & (step % 100 == 0):
            #         self.learning_rate = self.learning_rate / 10
            #         self.define_optimizer(self.learning_rate)
            #         print(f'Update learning rate to {self.learning_rate}')
            # past_loss = loss

    def predict_(self, data:pd.DataFrame, predict_size:int) -> Union[pd.Series, list]:
        """
        Predict data.
            data: the data you want to predict 
            predict_size: how many months you want to predict
        """
        self.data_setting(data, batch_size=len(data))
        predict_data, _ = next(self.training_data)
        predict_old_data = self.forward(predict_data)
        predict_old_data = predict_old_data.detach().numpy()
        predict_old_data = predict_old_data.reshape(len(predict_old_data),)

        predict_result = []
        counter = 0
        # prepare for predict
        latest_old_feature = data.loc[len(data) - 1, training_feature].astype('float').to_list() # Get the latest training features
        latest_old_feature.append(data.loc[len(data) - 1, target_col]) # Get the latest predict result for training features
        latest_old_feature.pop(0) # Remove the oldest training featrue
        while counter < predict_size:
            predict = self.forward(torch.from_numpy(np.array(latest_old_feature).reshape(1, self.feature_size, self.input_size))).detach().numpy()[0, 0]
            predict_result.append(predict)
            latest_old_feature.append(predict)
            latest_old_feature.pop(0)
            counter += 1
        return predict_old_data, predict_result
    
    def save_model(self, path:str) -> None:
        """
        Save model to specific path
            path: model path
        """
        torch.save(self.state_dict(), path)

    def load_model(self, path:str) -> nn.Module:
        """
        Load pytorch model from specific path.
            path: model path
        """
        self.load_state_dict(torch.load(path))
        self.double()

    def data_setting(self, data, batch_size=1):
        training_data = self.TrainingDataset(self, data)
        self.training_data = torch.utils.data.DataLoader(training_data, batch_size=batch_size ,num_workers=6, drop_last=False, shuffle=False)

    class TrainingDataset(Dataset):
        def __init__(self, model, data):
            self.model = model
            self.data = torch.tensor(np.array(data.loc[:, training_feature])).reshape(len(data), self.model.feature_size, self.model.input_size)
            self.target = torch.tensor(np.array(data.loc[:, target_col])).reshape(len(data), 1)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            return self.data[index], self.target[index]
        
if __name__ == "__main__":
    data = pd.read_csv('')
    monthly_data = difference_data(monthly_data)
    scaler, monthly_data = normalize_data(monthly_data)
    monthly_data = lag_data(monthly_data)
    
    model = LSTM_MODEL(input_size, hidden_size, num_layers)
    model.data_setting(monthly_data, batch_size)
    model.define_optimizer(learning_rate)
    model.define_loss_function()
    model.double()
    model.training_(training_step)
    model.save_model('./xx.pt')
    predict_old_data, predict_result = model.predict_(monthly_data, 100)

    # load model 
    # model = LSTM_MODEL(input_size, hidden_size, num_layers)
    # model = load_model('./xx.pt')