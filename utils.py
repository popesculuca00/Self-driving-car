import os
import torch


def get_batch_mask(commands):  # shape ( commands,  batch_size, params ) == (4 , batch_size, 3) 
    mask = torch.zeros(4, len(commands), 3)
    for i in range( len(commands) ):
        mask[ commands[i], i, :] = torch.ones(3)
    return mask

def save_model(model, epoch, path=None, optimizer=None):
    model.eval()
    torch.save(model.state_dict(), os.path.join(path, f"model_epoch_{epoch}.pth"))
    if optimizer:
        torch.save(optimizer.state_dict(), os.path.join(path, f"optimizer_epoch_{epoch}.pth"))
    model.train()

def load_model(model, model_path, optimizer_path=None):
    """
    Loads model and optimizer (if given) from state_dict
    """
    model.load_state_dict( torch.load(model_path))
    if optimizer_path:
        optim = torch.load(optimizer_path)
        return model, optim
    return model

def jit_compile_model(model):
    with torch.jit.optimized_execution(True):
        jitted_model = torch.jit.script(model)
    return jitted_model

