import params
from dataset import EGGDataset
from transformer_model import EGGPhrasePredictor
from cnn_model import CNNModel
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import params
import torch
from tqdm import tqdm
from dev import get_device

device = get_device()
model_type = "CNN"
# model_type = "Transformer"
if model_type == "CNN":
    tile_seq = False
else:
    tile_seq = True

def get_model(n_class):
    if model_type == "CNN":
        model = CNNModel(n_class=n_class).to(device)
    else:
        model = EGGPhrasePredictor(n_class=n_class, dmodel=params.D_MODEL).to(device)
    return model
def train():

    dataset = EGGDataset(tile_seq=tile_seq)
    n_class = dataset.get_num_class()
    model = get_model(n_class)
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    train_dt, test_dt = random_split(dataset, [0.8, 0.2], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE, num_workers=1,shuffle=True)
    test_dataloader = DataLoader(test_dt, batch_size=params.BATCH_SIZE,shuffle=False)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    min_test_loss = 1e6
    min_id = -1
    for epoch_id in range(params.N_EPOCH):
        model.train()
        for it, data in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            x, lb = data
            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            # print("X in", x.shape)
            prediction = model(x)
            loss = loss_function(prediction, lb.to(device))
            loss.backward()
            # if it % 10 == 0:
            #     print(it, loss)
            optimizer.step()
        true_test = []
        predicted_test = []
        print("Train last loss: ", loss)
        model.eval()
        for _, data in enumerate(test_dataloader):
            x, lb = data
            if model.type == "Transformer":
                x = x.transpose(1, 0)
            else:
                x = torch.unsqueeze(x, 1)
            x = x.float().to(device)
            # print("X in", x.shape)
            prediction = model(x)
            true_test.append(lb)
            predicted_test.append(prediction)

        true_test = torch.concat(true_test, dim=0).cpu()
        predicted_test = torch.concat(predicted_test, dim=0).cpu()
        test_loss = loss_function(predicted_test, true_test)
        if min_test_loss > test_loss:
            min_test_loss = test_loss
            min_id = epoch_id
        print("Error Test: ", test_loss, min_test_loss, epoch_id, min_id)

if __name__ == "__main__":
    train()
