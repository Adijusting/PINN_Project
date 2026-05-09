import torch

def calculate_physics_loss(model, inputs):
    """
    Inputs shape: [batch_size, 6] -> [lat(y), lon(x), time, elevation(z), precip(p), temp]"""
    
    # 1. Tell PyTorch we need to track gradients for inputs
    inputs.requires_grad_(True)
    
    # 2. Forward pass: get predictions
    predictions = model(inputs)
    h = predictions[:, 0:1]
    u = predictions[:, 1:2]
    v = predictions[:, 2:3]
    
    # #. Extract individual inputs for easier math
    y = inputs[:, 0:1]
    x = inputs[:, 1:2]
    t = inputs[:, 2:3]
    z = inputs[:, 3:4]
    p = inputs[:, 4:5]
    
    # 4. Calculate Gradients
    dh = torch.autograd.grad(h, inputs, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    dh_dy = dh[:, 0:1]
    dh_dx = dh[:, 1:2]
    dh_dt = dh[:, 2:3]
    
    du = torch.autograd.grad(u, inputs, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    du_dy = du[:, 0:1]
    du_dx = du[:, 1:2]
    du_dt = du[:, 2:3]
    
    dv = torch.autograd.grad(v, inputs, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dy = dv[:, 0:1]
    dv_dx = dv[:, 1:2]
    dv_dt = dv[:, 2:3]
    
    dz = torch.autograd.grad(z, inputs, grad_outputs=torch.ones_like(z), create_graph=True)[0]
    dz_dy = dz[:, 0:1]
    dz_dx = dz[:, 1:2]
    
    g = 9.81
    
    # 5. Build the differential equations
    # If network obeys physics, these equations will equal to 0
    
    # Mass conservation: dh/dt + h*du/dt + u*dh/dx + h*dv/dy + v*dh/dy - P = 0
    mass_residual = dh_dt + (h*du_dx + u*dh_dx) + (h*dv_dy + v*dh_dy) - p
    
    # Momentum X: du/dt + u*du/dx + v*du/dy + g*(dh/dx + dz/dx) = 0
    mom_x_residual = du_dt + u*du_dx + v*du_dy + g*(dh_dx + dz_dx)
    
    # Momentum Y: dv/dt + u*dv/dy + g*(dh/dy + dz/dy) = 0
    mom_y_residual = dv_dt + u*dv_dy + g*(dh_dy + dz_dy)
    
    # 6. Calculate FInal loss
    loss_mass = torch.mean(mass_residual**2) * 10000.0
    loss_mom_x = torch.mean(mom_x_residual**2)
    loss_mom_y = torch.mean(mom_y_residual**2)
    
    # 7. Negative Water Penalty
    # Water cannot be negative! If 'h' drops below 0, this creates a massive mathematical penalty.
    loss_negative_depth = torch.mean(torch.relu(-h)**2) * 10000.0
    
    total_physics_loss = loss_mass + loss_mom_x + loss_mom_y + loss_negative_depth
    
    return total_physics_loss

if __name__ == "__main__":
    from model import FloodPINN
    
    print("Testing Physics Loss Engine...")
    model = FloodPINN()
    
    # create 5 random data points
    dummy_inputs = torch.rand((5,6))
    loss = calculate_physics_loss(model, dummy_inputs)
    
    print("\nPhysics Loss Test")
    print(f"Inital Physics Violation (Loss): {loss.item():.4f}")
    print(f"If you see a number above, calculas engine is working")
    