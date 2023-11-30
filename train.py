import params
from dataset import EGGDataset
from model import EGGPhrasePredictor
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import params
import torch
from dev import get_device

device = get_device()

def train():
    dataset = EGGDataset()
    n_class = dataset.get_num_class()
    model = EGGPhrasePredictor(n_class=n_class, dmodel=params.D_MODEL).to(device)
    generator1 = torch.Generator().manual_seed(params.RD_SEED)
    train_dt, test_dt = random_split(dataset, [0.95, 0.05], generator=generator1)
    train_dataloader = DataLoader(train_dt, batch_size=params.BATCH_SIZE, num_workers=1,shuffle=True)
    test_dataloader = DataLoader(test_dt, batch_size=params.BATCH_SIZE)
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch_id in range(params.N_EPOCH):
        for it, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            x, lb = data
            x = x.transpose(1, 0)
            x = x.float().to(device)
            prediction = model(x)
            # print(prediction.shape, lb.shape)
            loss = loss_function(prediction, lb.to(device))
            loss.backward()
            if it % 10 == 0:
                print(it, loss)
            optimizer.step()


if __name__ == "__main__":
    train()
