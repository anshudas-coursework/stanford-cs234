import torch
import torch.nn as nn

def build_mlp(input_size, output_size, n_layers, size, type=None):
    """
    Args:
        input_size: int, the dimension of inputs to be given to the network
        output_size: int, the dimension of the output
        n_layers: int, the number of hidden layers of the network
        size: int, the size of each hidden layer
    Returns:
        An instance of (a subclass of) nn.Module representing the network.
    """
    if (n_layers==0): return nn.Linear(input_size,output_size)
    elif (type=='gru'):
        nn_list = [nn.Linear(input_size,size), nn.Tanh()]
        nn_list.extend([layer for i in range(n_layers-1) for layer in [nn.Linear(size,size), nn.Tanh()]])
        nn_list.append(nn.Linear(size,output_size))
        nn_list.append(nn.Tanh())
        return nn.Sequential(*nn_list)
    else:
        nn_list = [nn.Linear(input_size,size), nn.Tanh()]
        nn_list.extend([layer for i in range(n_layers-1) for layer in [nn.Linear(size,size), nn.Tanh()]])
        nn_list.append(nn.Linear(size,output_size))
        nn_list.append(nn.Tanh())
        return nn.Sequential(*nn_list)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def np2torch(x, cast_double_to_float=True):
    """
    Utility function that accepts a numpy array and does the following:
        1. Convert to torch tensor
        2. Move it to the GPU (if CUDA is available)
        3. Optionally casts float64 to float32 (torch is picky about types)
    """
    x = torch.from_numpy(x).to(device)
    if cast_double_to_float and x.dtype is torch.float64:
        x = x.float()
    return x
