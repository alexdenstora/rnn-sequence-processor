import torch
import torch.nn as nn
import torch.nn.functional as F

import pickle
import os

from torch.optim import AdamW
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

import matplotlib.pyplot as plt
import datetime
import random
import string
import wandb
from tqdm import tqdm

# Import our own files
from data.PoSData import Vocab, getUDPOSDataloaders
from models.PoSGRU import PoSGRU

use_cuda_if_avail = True
if use_cuda_if_avail and torch.cuda.is_available():
    device = "cuda"
elif use_cuda_if_avail and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

config = {
    "bs":256,   # batch size
    "lr":0.0005, # learning rate
    "l2reg":0.0000001, # weight decay
    "max_epoch":30,
    "layers": 2,
    "embed_dim":128,
    "hidden_dim":256,
    "residual":True
}


def main():

  # Get dataloaders
  train_loader, val_loader, _, vocab = getUDPOSDataloaders(config["bs"])

  vocab_size = vocab.lenWords()
  label_size = vocab.lenLabels()

  # Build model
  model = PoSGRU(vocab_size=vocab_size, 
                 embed_dim=config["embed_dim"], 
                 hidden_dim=config["hidden_dim"], 
                 num_layers=config["layers"],
                 output_dim=label_size,
                 residual=config["residual"])
  print(model)

  torch.compile(model)


  # Start model training
  train(model, train_loader, val_loader)




def train(model, train_loader, val_loader):
  os.makedirs('./chkpts', exist_ok=True)
  # Log our exact model architecture string
  config["arch"] = str(model)
  run_name = generateRunName()

  # Startup wandb logging
  wandb.login()
  wandb.init(project="UDPOS CS435 A6", name=run_name, config=config)

  # Move model to the GPU
  model.to(device)

  # Set up optimizer and our learning rate schedulers
  optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["l2reg"])
  warmup_epochs = config["max_epoch"]//10
  linear = LinearLR(optimizer, start_factor=0.25, total_iters=warmup_epochs)
  cosine = CosineAnnealingLR(optimizer, T_max = config["max_epoch"]-warmup_epochs)
  scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs])

  # checkpointing
  best_val = 0
  with open('./data/data.pckl', 'wb') as vocab:
    pickle.dump(train_loader.dataset.vocab, vocab)
  # Loss
  ###########################################
  #
  # Q5 TODO Loss
  #
  ###########################################
  entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

  # Main training loop with progress bar
  iteration = 0
  pbar = tqdm(total=config["max_epoch"]*len(train_loader), desc="Training Batches", unit="batch")
  for epoch in range(config["max_epoch"]):
    model.train()

    # Log LR
    wandb.log({"LR/lr": scheduler.get_last_lr()[0]}, step=iteration)

    for x, y, lens in train_loader:
      x = x.to(device)
      y = y.to(device)

      out = model(x)
      

      ###########################################
      #
      # Q5 TODO Loss
      #
      ###########################################
      loss = 0
      # change out shape from (Batch, T, and output_dim) to B x output_dim
      # input
      input = torch.flatten(input=out, start_dim=0, end_dim=1)
      # targets
      targets = torch.reshape(y, (-1, ))
      loss = entropy_loss(input, targets)

      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      
      ###########################################
      #
      # Q5 TODO Accuracy
      #
      ###########################################
      acc = 0

      predictions = torch.argmax(input=input, dim=-1) # find the predicted class
      matches = (predictions == targets) # find matches including the padding
      mask = (targets != -1) # create a mask to filter out -1 paddings
      real_tokens = matches[mask] # extract the real tokens (-1 == False, else True)
      # true_count = torch.sum(real_tokens) # count the number of Trues
      
      acc = real_tokens.float().mean() # convert to floats and then take the mean to get acc

      wandb.log({"Loss/train": loss.item(), "Acc/train": acc.item()}, step=iteration)
      pbar.update(1)
      iteration+=1

    val_loss, val_acc = evaluate(model, val_loader)
    wandb.log({"Loss/val": val_loss, "Acc/val": val_acc}, step=iteration)

    ###########################################
    #
    # Q6 TODO Checkpointing
    #
    ###########################################
   
    if val_acc > best_val:
      best_val = val_acc
      torch.save(model.state_dict(), "chkpts/" + run_name + "_epoch"+ str(epoch))
      torch.save(model.state_dict(), "chkpts/" + "best_checkpoint")

    # Adjust LR
    scheduler.step()

  wandb.finish()
  pbar.close()


def evaluate(model, loader):
  ###########################################
  #
  # Q6 TODO
  #
  ###########################################
  model.eval()
  running_loss = 0
  running_acc = 0
  nonpad = 0
  criterion = nn.CrossEntropyLoss(reduction="sum", ignore_index=-1)
  for x, y, lens in loader:
     x = x.to(device)
     y = y.to(device)
     out = model(x)
     loss = 0
     acc = 0
     
     # reshaping outputs to match input for calculating loss
     input = torch.flatten(input=out, start_dim=0, end_dim=1)
     targets = torch.reshape(y, (-1, ))

     # find predictions & remove padding
     predictions = torch.argmax(input=input, dim=-1)
     matches = (predictions == targets)
     mask = (targets != -1)
     real_tokens = matches[mask]
     
     # calculate the nonpad tokens for this batch
     batch_nonpad = mask.sum().item()
     nonpad += batch_nonpad # add that to the global nonpad

     loss = criterion(input, targets)
     batch_correct = real_tokens.sum().item() # sum the number of correct predictions
     
     running_loss += loss.item() # add the loss to the running loss
     running_acc += batch_correct # add the total correct to the global accuracy count

  return running_loss/nonpad, running_acc/nonpad # calculate the total loss/acc based on the total nonpad tokens

def generateRunName():
  random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
  now = datetime.datetime.now()
  run_name = ""+random_string+"_UDPOS"
  return run_name



main()
