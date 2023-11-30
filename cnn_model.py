import torch
from torch import nn


class CNNModel(nn.Module):
    def __init__(self, n_class, n_base=16, n_conv=8):
        super().__init__()
        self.n_class = n_class
        self.type= "CNN"

        self.layer1 = nn.Sequential(nn.Conv1d(1, n_base * 3, kernel_size=11, stride=4, padding=0),
                                    nn.BatchNorm1d(n_base * 3),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2))

        self.layer2 = nn.Sequential(nn.Conv1d(n_base * 3, n_base * 8, kernel_size=5, stride=1, padding=2),
                                    nn.BatchNorm1d(n_base * 8),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2))

        self.layer3 = nn.Sequential(nn.Conv1d(n_base * 8, n_base * 10, kernel_size=3, stride=1, padding=2),
                                    nn.BatchNorm1d(n_base * 10),
                                    nn.ReLU(),
                                    # nn.MaxPool1d(kernel_size=3, stride=2)
                                    )

        self.layer4 = nn.Sequential(nn.Conv1d(n_base * 10, n_base * 8, kernel_size=3, stride=1, padding=2),
                                    nn.BatchNorm1d(n_base * 8),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2)
                                    )

        self.layer5 = nn.Sequential(nn.Conv1d(n_base * 8, n_base * 6, kernel_size=3, stride=1, padding=2),
                                    nn.BatchNorm1d(n_base * 6),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=3, stride=2)
                                    )
        self.fc1 = nn.Sequential(nn.Dropout(0.05), nn.Linear(1536, 320), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(320, n_class))

    def forward(self, x):
        # print("X", x.shape)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
