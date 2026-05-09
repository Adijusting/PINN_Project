import torch
import torch.nn as nn

class FloodPINN(nn.Module):
    def __init__(self, num_layers=6, neurons_per_layer=64):
        super(FloodPINN, self).__init__()
        
        # We have 6 input features: [lat, lon, time, elevation, precip, temp]
        input_dim = 6
        
        # 3 output features: [depth(h), velocity_x(u), velocity_y(v)]
        output_dim = 6
        
        # 1. Input Layers
        layers = [nn.Linear(input_dim, neurons_per_layer), nn.Tanh()]
        # 2. Hidden layers
        # We use Tanh as it is smoothly diffrentiable
        for _ in range(num_layers-1):
            layers.append(nn.Linear(neurons_per_layer, neurons_per_layer))
            layers.append(nn.Tanh())
            
        # 3. Output layer
        layers.append(nn.Linear(neurons_per_layer, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        x is a tensor containing out input: [lat, lon, time, elevation, precip, temp]
        returns a tensor with our predictions: [h,u,v]
        """
        return self.network(x)
    
if __name__ == "__main__":
    print("Initializing Flood PINN architecture...")
    model=FloodPINN()
    print(model)
        
    dummy_input = torch.tensor([[50.5, 7.0, 12.0, 250.0, 0.05, 298.0]], dtype=torch.float32)
        
    dummy_output = model(dummy_input)
    print("\nNetwork plumbing test")
    print(f"Input shape (1 data point, 6 features): {dummy_input.shape}")
    print(f"Output shape (1 data point, 3 predictions): {dummy_output.shape}")
    print(f"RAW untrained predictions [h,u,v]: {dummy_output.detach().numpy()}")