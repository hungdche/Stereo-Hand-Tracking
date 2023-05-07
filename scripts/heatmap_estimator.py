import torch
import torch.nn.functional as F

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
        batch_size = x.shape[0]
        x_96 = x
        x_48 = F.interpolate(x_96, size=(48, 48), mode='bilinear')
        x_24 = F.interpolate(x_96, size=(24, 24), mode='bilinear')
   
        
        cnn1 = self.bank1(x_96)
        cnn2 = self.bank2(x_48)
        cnn3 = self.bank3(x_24)
        
        flat1 = cnn1.view(batch_size, -1)
        flat2 = cnn1.view(batch_size, -1)
        flat3 = cnn1.view(batch_size, -1)
        
        flattened = torch.cat((flat1, flat2, flat3), dim=1)
        
        result = self.fc(flattened)
        reshaped_result = torch.reshape(result, (batch_size, 21, 18, 18))
        return reshaped_result