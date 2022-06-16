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
        self.embedding_type = 'None'

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
        # if config.model.sigmoid_last:
        #     self.mlp.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*self.mlp)

    def log_energy(self, x, t):
        t = t.unsqueeze(-1)
        inpt = torch.cat([x, t], dim=1)
        out = self.mlp(inpt)
        return out

    def energy(self, x ,t):
        return torch.exp(self.log_energy(x,t))

    def score(self, x, t):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            out = self.log_energy(x, t)
            gradients = torch.autograd.grad(outputs=out, inputs=x,
                                    grad_outputs=torch.ones(out.size()).to(self.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
        return gradients

    def trace_hessian_log_energy(self, x, t, full=False, exact=False):
        if not exact:
            with torch.enable_grad():
                x = x.requires_grad_(True)
                score = self.score(x,t)
                grads = []
                for i, v in enumerate(torch.eye(x.shape[1], device=self.device)):
                    gradients = torch.autograd.grad(outputs=score, inputs=x,
                                        grad_outputs=v.repeat(x.shape[0],1),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0][:,i]
                    grads.append(gradients)
                grads = torch.stack(grads,dim=1)
                return torch.sum(grads, dim=1)

                #x = x.requires_grad_(True)
                #inputs =  torch.cat([x, t[...,None]], dim=1)
                #log_energy_t = lambda inpt: self.log_energy(inpt[None, :2], inpt[2:]).squeeze()
                #hessian = torch.stack([torch.autograd.functional.hessian(log_energy_t, inpt)[:-1,:-1] for inpt in inputs])
                #trace = torch.stack([torch.trace(hess) for hess in hessian])
                #if full:
                #    return hessian
                #else:
                #    return trace
        # else:
        #     score = self.score(x,t)
        #     epsilon = torch.randn_like(x)
        #     with torch.enable_grad():
        #         x = x.requires_grad_(True)
        #         eps_transpose_H = torch.autograd.grad(outputs=score, inputs=x,
        #                             grad_outputs=epsilon,
        #                             create_graph=True, retain_graph=True, only_inputs=True)[0]
            

    def time_derivative_log_energy(self, x, t):
        with torch.enable_grad():
            t = t.requires_grad_(True)
            out = self.log_energy(x, t)
            gradients = torch.autograd.grad(outputs=out, inputs=t,
                                    grad_outputs=torch.ones(out.size()).to(self.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
        return gradients.squeeze()


    def forward(self, x, t):
        return self.score(x,t)


@utils.register_model(name='fcn_compound')
class FCN_PotentialCompound(FCN_Potential):

    def __init__(self, config):
        super(FCN_Potential, self).__init__()

        state_size = config.model.state_size
        hidden_layers = config.model.hidden_layers
        hidden_nodes = config.model.hidden_nodes
        dropout = config.model.dropout
        self.embedding_type = 'None'

        input_size = state_size + 1 #+1 because of the time dimension.
        output_size = 1

        # fokker planck net
        self.mlp_fp = nn.ModuleList()
        self.mlp_fp.append(nn.Linear(input_size, hidden_nodes))
        self.mlp_fp.append(nn.Dropout(dropout)) #addition
        self.mlp_fp.append(nn.ELU())

        for _ in range(hidden_layers):
            self.mlp_fp.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp_fp.append(nn.Dropout(dropout)) #addition
            self.mlp_fp.append(nn.ELU())
        
        self.mlp_fp.append(nn.Linear(hidden_nodes, output_size))
        self.mlp_fp = nn.Sequential(*self.mlp_fp)


        # correction net
        self.mlp_corrector = nn.ModuleList()
        self.mlp_corrector.append(nn.Linear(input_size, hidden_nodes))
        self.mlp_corrector.append(nn.Dropout(dropout)) #addition
        self.mlp_corrector.append(nn.ELU())

        for _ in range(hidden_layers):
            self.mlp_corrector.append(nn.Linear(hidden_nodes, hidden_nodes))
            self.mlp_corrector.append(nn.Dropout(dropout)) #addition
            self.mlp_corrector.append(nn.ELU())
        
        self.mlp_corrector.append(nn.Linear(hidden_nodes, output_size))
        self.mlp_corrector = nn.Sequential(*self.mlp_fp)


    def log_energy(self, x, t, weight_fp=1, weight_corerctor=1):
        t = t.unsqueeze(-1)
        inpt = torch.cat([x, t], dim=1)
        out = weight_fp*self.mlp_fp(inpt) + weight_corerctor*self.mlp_corrector(inpt)
        return out

    def score(self, x, t, weight_fp=1, weight_corerctor=1):
        with torch.enable_grad():
            x = x.requires_grad_(True)
            out = self.log_energy(x, t, weight_fp, weight_corerctor)
            gradients = torch.autograd.grad(outputs=out, inputs=x,
                                    grad_outputs=torch.ones(out.size()).to(self.device),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
            gradients = gradients.view(gradients.size(0), -1)
        return gradients
