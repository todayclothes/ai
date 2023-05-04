import torch.nn as nn
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.l1 = nn.Linear(7, 512, bias = True)
        self.l2 = nn.Linear(512, 256, bias = True)
        self.l3 = nn.Linear(256, 64, bias = True)
        self.l4 = nn.Linear(64, 80, bias = True)
        
        self.relu = nn.ReLU()
        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                m.bias.data.zero_()
                
    def forward(self, x):
        x = self.l1(x)
        x = self.relu(x)
        
        x = self.l2(x)
        x = self.relu(x)
        
        x = self.l3(x)
        x = self.relu(x)
        
        x = self.l4(x)
        
        return x