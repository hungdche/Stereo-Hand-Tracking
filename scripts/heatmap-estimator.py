import torch

class HeatmapEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Input dimensions: 1x96x96
        self.bank1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4),
            torch.nn.Conv2d(16, 32, 6),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Input dimensions: 1x48x48
        self.bank2 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(16, 32, 5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Input dimensions: 1x24x24
        self.bank3 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, 4),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        
        # Final FC layers
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(7776, 6804),
            torch.nn.ReLU(),
            torch.nn.Linear(6804, 6804)
        )
        
    def forward(self, x):
        