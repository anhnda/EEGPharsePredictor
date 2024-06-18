import torch
from torch import nn



class MNAPooling(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)
        self.avg_pooling = nn.AvgPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        mx = self.max_pooling(x)
        avg = self.avg_pooling(x)
        return torch.concat([mx, avg], dim=1)


class BiMaxPooling(nn.Module):
    def __init__(self, kernel_size=3, stride=2):
        super().__init__()
        self.max_pooling = nn.MaxPool1d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        mx1 = self.max_pooling(x)
        mx2 = self.max_pooling(-x)
        return torch.concat([mx1, mx2], dim=1)


def get_dim(dim, flag):
    return dim * flag


class CNNModel3C3Out(nn.Module):
    def __init__(self, n_class, n_base=16, flag=1, out_collapsed=True, n_conv=8):
        super().__init__()
        self.n_class = n_class
        self.type = "CNN3C_Out"
        self.flag = flag
        self.chain1_layers = nn.ModuleList()
        self.chain2_layers = nn.ModuleList()
        self.chain3_layers = nn.ModuleList()
        self.chains = [self.chain1_layers, self.chain2_layers, self.chain3_layers]
        self.out_collapsed = out_collapsed
        base_dim = 1536

        for i in range(3):
            layer1 = nn.Sequential(nn.Dropout(0.1),
                                   nn.Conv1d(1, n_base * 3, kernel_size=11, stride=4, padding=0),
                                   # nn.BatchNorm1d(n_base * 3),
                                   nn.LeakyReLU(),
                                   # nn.MaxPool1d(kernel_size=3, stride=2)
                                   BiMaxPooling(kernel_size=3, stride=2)
                                   )

            layer2 = nn.Sequential(nn.Conv1d(n_base * 3 * 2, n_base * 8, kernel_size=5, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 8),
                                   nn.LeakyReLU(),
                                   BiMaxPooling(kernel_size=3, stride=2))

            layer3 = nn.Sequential(nn.Conv1d(n_base * 8 * 2, n_base * 20, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 10),
                                   nn.LeakyReLU(),
                                   BiMaxPooling(kernel_size=3, stride=2)
                                   )

            layer4 = nn.Sequential(nn.Conv1d(n_base * 20 * 2, n_base * 8, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 8),
                                   nn.LeakyReLU(),
                                   BiMaxPooling(kernel_size=3, stride=2)
                                   )

            layer5 = nn.Sequential(nn.Conv1d(n_base * 8 * 2, n_base * 6, kernel_size=3, stride=1, padding=2),
                                   # nn.BatchNorm1d(n_base * 6),
                                   nn.LeakyReLU(),
                                   BiMaxPooling(kernel_size=3, stride=2)
                                   )
            self.chains[i].append(layer1)
            self.chains[i].append(layer2)
            self.chains[i].append(layer3)
            self.chains[i].append(layer4)
            self.chains[i].append(layer5)

            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(2304, 320), nn.ReLU())
            fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(get_dim(base_dim, self.flag), 320), nn.LeakyReLU())
            self.chains[i].append(fc1)
            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(1536, 320), nn.ReLU())

            # self.fc1 = nn.Sequential(nn.Dropout(0.1), nn.Linear(768, 320), nn.ReLU())

        self.fc2 = nn.Sequential(nn.Linear(320 * 3, n_class * 3))

    def forward(self, x):
        # print("X", x.shape)
        xis = []
        for i in range(3):
            xi = x[:, :, i, :]
            xi = torch.squeeze(xi, 2)
            for j, jlayer in enumerate(self.chains[i]):
                if j < len(self.chains[i]) - 1:
                    xi = jlayer(xi)
                else:
                    xi = xi.reshape(xi.size(0), -1)
                    xi = jlayer(xi)
                    xis.append(xi)
        out = torch.concat(xis, dim=-1)
        out = self.fc2(out)
        if self.out_collapsed:
            out = out.reshape((out.shape[0], -1, 3))

        return out
