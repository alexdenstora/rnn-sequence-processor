import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class PoSGRU(nn.Module) :

    def __init__(self, vocab_size=1000, embed_dim=16, hidden_dim=16, num_layers=2, output_dim=10, residual=True) :
      super().__init__()
      ###########################################
      #
      # Q4 TODO
      #
      ###########################################
      self.residual_flag = residual
      self.num_layers = num_layers
      # self.layers = nn.Sequential() # container for the grus

      # embed layer - num_embeddings = vocab_size, embedding_dim = embed_dim, padding_idx
      self.embed = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=None)
      # linear layer - maps embed dimension -> hidden_dim, num_layers
      self.linear = nn.Linear(in_features=embed_dim, out_features=hidden_dim)

      # Bidirectional GRUs - hidden_dim = hidden_dim/2 (maybe batch first), originally num_layers=num_layers
      # gru module
      self.grus = nn.ModuleList([
         nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim//2, num_layers=1, batch_first=True, bidirectional=True) for _ in range(num_layers)])

      # classifier module:
      self.classifier = nn.Sequential(
         nn.Linear(in_features=hidden_dim, out_features=hidden_dim), # Linear Layer - hidden_dim
         nn.GELU(), # GELU
         nn.Linear(in_features=hidden_dim, out_features=output_dim) # Linear Layer - output_dim
      )

  
    def forward(self, x):
      ###########################################
      #
      # Q4 TODO
      #
      ###########################################
      x = self.embed(x)
      x = self.linear(x)

      for gru in self.grus:
         residual = x
         x, _ = gru(x)
         if self.residual_flag:
            x = x + residual

      return self.classifier(x)
