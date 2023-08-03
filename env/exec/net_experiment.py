#! usr/bin/env python3


import torch
import torch.nn as nn
from torch.func import vmap,jacrev

from pyoptmat import ode, experiments, utility


import pdb

import numpy as np
import scipy
import matplotlib.pyplot as plt

from tqdm import tqdm

import xarray as xr

if torch.cuda.is_available():
    dev = "cuda:1"
else:
    dev = "cpu"
device = torch.device(dev)
# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)


class Network(nn.Module):
    '''
    Simple neural network with 5 inputs and 5 outputs
    '''
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        
        network = nn.Sequential(
            nn.Linear(5,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3),
            nn.LeakyReLU(),
            nn.Linear(3,3)
        ).to(device)
        self.network = network

        self.n_internal = self.network[-1].weight.shape[1] - 1        
        self.nsize = self.n_internal + 1

    def rate(self, t, y, erate, T):
        '''
        Manually eval the network
        '''
        
        x = torch.cat((y,erate,T),dim=-1)


        fc1 = self.network[1](torch.einsum('ij,...j', self.network[0].weight, x) + self.network[0].bias)
        fc2 = self.network[3](torch.einsum('ij,...j', self.network[2].weight, fc1)+ self.network[2].bias)
        fc3 = self.network[5](torch.einsum('ij,...j', self.network[4].weight, fc2)+ self.network[4].bias)
        fc4 = self.network[7](torch.einsum('ij,...j', self.network[6].weight, fc3)+ self.network[6].bias)
        fc5 = self.network[9](torch.einsum('ij,...j', self.network[8].weight, fc4)+ self.network[8].bias)
        fc6 = self.network[11](torch.einsum('ij,...j', self.network[10].weight, fc5)+ self.network[10].bias)
        fc7 = self.network[13](torch.einsum('ij,...j', self.network[12].weight, fc6)+ self.network[12].bias)
        fc8 = self.network[15](torch.einsum('ij,...j', self.network[14].weight, fc7)+ self.network[14].bias)
        fc9 = self.network[17](torch.einsum('ij,...j', self.network[16].weight, fc8)+ self.network[16].bias)
        fc10 = torch.einsum('ij,...j', self.network[18].weight, fc9)+ self.network[18].bias

        return fc10


    def forward(self, t, y, erate, T):
        '''
        Forward pass of the network
          Inputs
            t: time (1000,1)
            y: state (1000,3)
            erate: strain rate (75,1000)
            T: some random variable (75,1000)
          Returns
            ydot: derivative of the state (1000,5)
        '''
        erate = erate.unsqueeze(dim=-1)
        T = T.unsqueeze(dim=-1)
        y = y.permute(1,2,0)

        x = torch.cat((y,erate,T),dim=-1)
        
        return self.rate(t,y,erate,T), vmap((jacrev(self.network)))(x)


def normalize(inp):
    min_v = torch.min(inp.reshape((inp.shape[0],) + (-1,)), dim = -1)[0]
    max_v = torch.max(inp.reshape((inp.shape[0],) + (-1,)), dim = -1)[0]
    return (inp - min_v[...,None,None]) / (max_v[...,None,None] - min_v[...,None,None])


class StrainBasedModel(nn.Module):
    """
    Provides the strain rate form

    Args:
      model:        base InelasticModel
      erate_fn:     erate(t)
      T_fn:         T(t)
    """

    def __init__(self, model, erate_fn, T_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.erate_fn = erate_fn
        self.T_fn = T_fn

    def forward(self, t, y):
        """
        Strain rate as a function of t and state

        Args:
            t:  input times
            y:  input state
        """
        return self.model(t, y, self.erate_fn(t), self.T_fn(t))[
            :2
        ]  # Don't need the extras



def solve_strain(times, strains, temperatures, model):
        """
        Solve for either strain or stress control at once

        Args:
          times:          input times, (ntime,nexp)
          temperatures:   input temperatures (ntime,nexp)
          idata:          input data (ntime,nexp)
          control:        signal for stress/strain control (nexp,)
        """
        
        
        times = times.permute(1,2,0).squeeze(dim=-1)
        strains = strains.permute(1,2,0).squeeze(dim=-1)
        temperatures = temperatures.permute(1,2,0).squeeze(dim=-1)
        
        strain_rates = torch.cat(
            (
                torch.zeros(1, strains.shape[1], device=strains.device),
                (strains[1:] - strains[:-1]) / (times[1:] - times[:-1]),
            )
        )
        
        # Likely if this happens dt = 0
        strain_rates[torch.isnan(strain_rates)] = 0

        erate_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, strain_rates
        )
        temperature_interpolator = utility.ArbitraryBatchTimeSeriesInterpolator(
            times, temperatures
        )

        init = torch.zeros(
            times.shape[1], 2 + model.nsize, device=device
        )
        init[:, -1] = 0.
        
        emodel = StrainBasedModel(model, erate_interpolator, temperature_interpolator)

        return ode.odeint_adjoint(emodel, init, times)


if __name__ == "__main__":

    # Load in the data
    input_data = xr.open_dataset("data.nc")
    data, results, cycles, types, control = experiments.load_results(
            input_data, device = device)

    data = normalize(data)
    results = normalize(results.unsqueeze(0)).squeeze(0)

    net = Network()

    # Define the parameters 
    t = data[0]
    stress = results.unsqueeze(dim = 0)
    erate = data[-1]
    T = data[2]

    latent_states = torch.ones((2,75,1000)) * 1e-8
    latent_states = latent_states.to(device)

    
    y = torch.cat((stress, latent_states), dim = 0) 
    y0 = torch.zeros((75,1000,3)).to(device)

    ydot, jac = net(t,y,erate,T)
    t0 = t.reshape(75,1000,1)

    ydot = solve_strain(t, erate, T, net)


    t_np = t.detach().numpy()
    y_np = ydot.detach().numpy()
    
    plt.plot(t_np,y_np[:,0])
    plt.savefig('ydot.png')


