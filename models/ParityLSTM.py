from torch import nn
import torch

class ParityLSTM(nn.Module) :

    def __init__(self, hidden_dim=16):
        super().__init__()
        ###########################################
        #
        # Q2 TODO
        #
        ###########################################
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=2)


    def forward(self, x, x_lens):
        ###########################################
        #
        # Q2 TODO
        #
        ###########################################
        # for each batch element i, collect the hidden state at time x_lens[i] - 1, this results in B x hidden_dim tensor
        # pass this through the linear layer to produce B x 2 and return that output
        hidden_states, state = self.lstm(x)
        batch_indices = torch.arange(x.shape[0])
        time_indices = x_lens - 1
        collection = hidden_states[batch_indices, time_indices]
        out = self.linear(collection)
        return out
    
