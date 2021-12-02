import torch.nn as nn
import torch
from . import utils
import pytorch_lightning as pl

@utils.register_model(name='fcn_potential')
class FCN_Potential(pl.LightningModule):
    def __init__(self, config): 
        super(FCN_Potential, self).__init__()
        state_size = config.model.state_size
        hidden_layers = config.model.hidden_layers
        hidden_nodes = config.model.hidden_nodes
        dropout = config.model.dropout

        input_size = state_size + 1 #+1 because of the time dimension.
        output_size = 1

        self.mlp = nn.ModuleList()
        self.mlp.append(nn.Linear(input_size, hidden_nodes))
        self.mlp.append(nn.Dropout(dropout)) #addition
        self.mlp.append(nn.ELU())

        for _ in range(hidden_layers):
            self.mlp.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp.append(nn.Dropout(dropout)) #addition
            self.mlp.append(nn.ELU())
        
        self.mlp.append(nn.Linear(hidden_nodes, output_size))
        self.mlp = nn.Sequential(*self.mlp)
             
    def forward(self, x, t):
        torch_grad_enabled =torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        x = x.clone().detach()
        x.requires_grad=True
        x_shape = x.shape
        t_shape = t.shape
        if len(x_shape)==2:
            #x_shape = (batchsize, state_size) --> sampling process - reverse SDE
            t = t.unsqueeze(-1)
            inpt = torch.cat([x, t], dim=1)
            out = self.mlp(inpt)
            gradients = torch.autograd.grad(outputs=out, inputs=x,
                                  grad_outputs=torch.ones(out.size()).to(self.device),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
            torch.set_grad_enabled(torch_grad_enabled)
            #return out
            return gradients
        else:
            raise NotImplementedError




        