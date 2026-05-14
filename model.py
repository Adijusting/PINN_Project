import torch
import torch.nn as nn

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        
        # --- THE NORMALIZATION SHIELD ---
        # We pre-program the approximate averages and ranges of the Ahr Valley.
        # Order: [Lat, Lon, Time, Elev, Precip, Init_Depth, Friction]
        self.register_buffer('input_mean', torch.tensor([50.55, 7.00, 36.0, 300.0, 0.001, 290.0, 0.05]))
        self.register_buffer('input_std', torch.tensor([0.15, 0.20, 36.0, 200.0, 0.010, 1.0, 0.05]))
        
        self.network = nn.Sequential(
            nn.Linear(7, 128),  
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 3) # 3 OUTPUTS: h, u, v
        )

    def forward(self, x):
        # 1. Scale the raw physical inputs down to manageable sizes (-1 to 1)
        x_normalized = (x - self.input_mean) / self.input_std
        
        # 2. Feed the balanced numbers into the network's brain
        return self.network(x_normalized)