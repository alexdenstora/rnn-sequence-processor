import torch
import pickle

from models.PoSGRU import PoSGRU
def main():
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

    file_path = './data/data.pckl'
    with open(file_path, 'rb') as file:
        vocab = pickle.load(file)

    label_size = vocab.lenLabels()
    model = PoSGRU( # build model
        vocab_size=vocab.lenWords(),
        embed_dim=config["embed_dim"], 
        hidden_dim=config["hidden_dim"], 
        num_layers=config["layers"],
        output_dim=label_size,
        residual=config["residual"]
        )
    path = './chkpts/best_checkpoint'
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    sequence = input()
    tokens = sequence.split()

    numeric_sequence = []
    for token in tokens:
        numeric_sequence.append(vocab.word2idx.get(token, vocab.word2idx['<UNK>']))


    tensor = torch.tensor(numeric_sequence)
    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(device)

    out = model(tensor)
    out = torch.argmax(out, dim=-1)
    out = out[0].cpu().tolist()
    print(out)
    
    output_sequence = []
    for index in out:
        output_sequence.append(vocab.idx2label.get(index))
    print(tokens)
    print(output_sequence)

if __name__ == "__main__":
    main()