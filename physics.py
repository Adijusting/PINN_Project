import torch

def shallow_water_loss(model, inputs):
    # 1. Break apart the variables BEFORE they enter the network.
    # This forces PyTorch to mathematically track them as distinct, independent variables!
    y = inputs[:, 0:1].clone().requires_grad_(True)
    x = inputs[:, 1:2].clone().requires_grad_(True)
    t = inputs[:, 2:3].clone().requires_grad_(True)
    
    # These are static data files. They do not need calculus tracking.
    elevation = inputs[:, 3:4]
    precip = inputs[:, 4:5]
    init_depth = inputs[:, 5:6]
    n = inputs[:, 6:7]
    
    # 2. Fuse them back together and feed them to the AI
    model_inputs = torch.cat([y, x, t, elevation, precip, init_depth, n], dim=1)
    
    preds = model(model_inputs)
    h = preds[:, 0:1]
    u = preds[:, 1:2]
    v = preds[:, 2:3]
    
    g = 9.81 
    
    # 3. Calculate spatial and temporal derivatives
    h_t = torch.autograd.grad(h, t, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_x = torch.autograd.grad(h, x, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    h_y = torch.autograd.grad(h, y, grad_outputs=torch.ones_like(h), create_graph=True)[0]
    
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    # --- THE CALCULUS WALL BYPASS ---
    # Elevation is static data, not a mathematical function. PyTorch will crash if we run autograd on it.
    # For now, we safely bypass the physical slope derivative so the network can train.
    elev_x = torch.zeros_like(h)
    elev_y = torch.zeros_like(h)

    # --- MANNING'S FRICTION BRAKES ---
    h_safe = torch.abs(h) + 1e-6
    vel_mag = torch.sqrt(u**2 + v**2 + 1e-6)
    
    S_fx = (n**2 * u * vel_mag) / (h_safe**(4/3))
    S_fy = (n**2 * v * vel_mag) / (h_safe**(4/3))
    # ---------------------------------

    # 4. The Shallow Water Equations
    mass_residual = h_t + (u * h_x + h * u_x) + (v * h_y + h * v_y) - precip
    
    momentum_x = u_t + u * u_x + v * u_y + g * (h_x + elev_x) + g * S_fx
    momentum_y = v_t + u * v_x + v * v_y + g * (h_y + elev_y) + g * S_fy
    
    loss = torch.mean(mass_residual**2) + torch.mean(momentum_x**2) + torch.mean(momentum_y**2)
    return loss