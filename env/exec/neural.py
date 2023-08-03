#!/usr/bin/env python3

import sys
sys.path.append('../../..')

import xarray as xr
import torch
import pyro
import matplotlib.pyplot as plt

import tqdm

from pyoptmat import optimize, experiments, models
from pyoptmat.utility import mbmm, macaulay

from functorch import vmap, jacrev

# Use doubles
torch.set_default_tensor_type(torch.DoubleTensor)

# Select device to run on
if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

activation = torch.nn.ReLU()

def normalize(inp):
    min_v = torch.min(inp.reshape((inp.shape[0],) + (-1,)), dim = -1)[0]
    max_v = torch.max(inp.reshape((inp.shape[0],) + (-1,)), dim = -1)[0]
    
    return (inp - min_v[...,None,None]) / (max_v[...,None,None] - min_v[...,None,None])

def network(y, erate, T, E, n, w_start, b_start, w_int, b_int, w_end, b_end):
    """
        Simple deep network example
    """
    x = torch.cat([y, erate.unsqueeze(-1), T.unsqueeze(-1)], dim = -1).unsqueeze(-1)

    state = activation(mbmm(w_start, x) + b_start.unsqueeze(-1))
    
    for w,b in zip(w_int, b_int):
        state = activation(mbmm(w, state) + b.unsqueeze(-1))

    state = mbmm(w_end, state) + b_end.unsqueeze(-1)

    state = state.squeeze(-1)

    return torch.cat([
        torch.exp(E) * (erate.unsqueeze(-1) - macaulay((torch.abs(y[0:1]) * (1+state[0:1]/100000)) ** torch.exp(n))*torch.sign(y[0:1])),
        state[1:]/10000.0
        ], dim = -1)

class Model(torch.nn.Module):
    def __init__(self, E, n, w_start, b_start, w_int, b_int, w_end, b_end):
        super().__init__()
        
        self.E = E
        self.n = n

        self.w_start = w_start
        self.b_start = b_start
        self.w_int = w_int
        self.b_int = b_int
        self.w_end = w_end
        self.b_end = b_end

        self.nhist = w_start.shape[-1] - 4

    def _expand(self, t, p):
        return p.expand(t.shape[:1] + p.shape)

    def forward(self, t, y, erate, T):
        """Evaluate the rate of the state plus the Jacobian terms

        """
        rate = vmap(vmap(network))(y, erate, T,
                       self._expand(t, self.E),
                       self._expand(t, self.n),
                       self._expand(t, self.w_start), self._expand(t, self.b_start),
                       self._expand(t, self.w_int), self._expand(t, self.b_int),
                       self._expand(t, self.w_end), self._expand(t,self.b_end))
        J1 = vmap(vmap(jacrev(network, argnums = 0)))(y, erate, T, 
                                                      self._expand(t, self.E),
                                                      self._expand(t, self.n),
                                                      self._expand(t, self.w_start), self._expand(t, self.b_start),
                                                      self._expand(t, self.w_int), self._expand(t, self.b_int),
                                                      self._expand(t, self.w_end), self._expand(t,self.b_end))
        J2 = vmap(vmap(jacrev(network, argnums = 1)))(y, erate, T, 
                                                      self._expand(t, self.E),
                                                      self._expand(t, self.n),
                                                      self._expand(t, self.w_start), self._expand(t, self.b_start),
                                                      self._expand(t, self.w_int), self._expand(t, self.b_int),
                                                      self._expand(t, self.w_end), self._expand(t,self.b_end))
        J3 = vmap(vmap(jacrev(network, argnums = 2)))(y, erate, T, 
                                                      self._expand(t, self.E),
                                                      self._expand(t, self.n),
                                                      self._expand(t, self.w_start), self._expand(t, self.b_start),
                                                      self._expand(t, self.w_int), self._expand(t, self.b_int),
                                                      self._expand(t, self.w_end), self._expand(t,self.b_end))

        return rate, J1, J2.squeeze(-1), J3.squeeze(-1)

if __name__ == "__main__":
    # Time chunking
    time_chunk_size = 10

    # Load in the data
    input_data = xr.open_dataset("data.nc")
    data, results, cycles, types, control = experiments.load_results(
            input_data, device = device)

    data = normalize(data)
    results = normalize(results.unsqueeze(0)).squeeze(0)

    # Number of tests
    ntests = data.shape[2]
    
    # How many history variables to maintain
    nhist = 5

    # Size of the inner layers
    sinner = 10

    # Number of inner layers
    ninner = 3

    def maker(E, n, w_start, b_start, w_int, b_int, w_end, b_end, **kwargs):
        rmodel = Model(E, n, w_start, b_start, w_int, b_int, w_end, b_end)
        return models.ModelIntegrator(rmodel, **kwargs, block_size = time_chunk_size)
    
    # Setup priors
    names = ["E", "n", "w_start", "b_start", "w_int", "b_int", "w_end", "b_end"]

    loc_loc_priors = [
            torch.log(torch.tensor([10.0], device = device)),
            torch.log(torch.tensor([6.0], device = device)),
            torch.rand((3 + nhist + sinner, 3 + nhist), device = device),
            torch.rand((3 + nhist + sinner,), device = device),
            torch.rand((ninner, 3 + nhist + sinner, 3 + nhist + sinner), device = device),
            torch.rand((ninner, 3 + nhist + sinner), device = device),
            torch.rand((1 + nhist, 3 + nhist + sinner), device = device),
            torch.rand((1 + nhist,), device = device)
            ]
    loc_scale_priors = [0.1 * torch.abs(l) for l in loc_loc_priors]
    scale_scale_priors = [0.1 * torch.abs(l) for l in loc_loc_priors]

    eps_prior = torch.tensor(5.0, device = device)
    
    smodel = optimize.HierarchicalStatisticalModel(
            lambda *args, **kwargs: maker(*args, **kwargs),
            names, loc_loc_priors, loc_scale_priors, scale_scale_priors, eps_prior,
            include_noise = True)
    
    # Get the guide
    guide = smodel.make_guide()

    # 5) Setup the optimizer and loss
    lr = 1.0e-3
    g = 1.0
    niter = 1000
    num_samples = 1

    optimizer = pyro.optim.ClippedAdam({"lr": lr})

    ls = pyro.infer.Trace_ELBO(num_particles=num_samples)

    svi = pyro.infer.SVI(smodel, guide, optimizer, loss=ls)

    # Actually infer
    t = tqdm.tqdm(range(niter), total=niter, desc="Loss:    ")
    loss_hist = []
    for i in t:
        loss = svi.step(data.to(device), cycles.to(device), types.to(device),
                control.to(device), results.to(device))
        loss_hist.append(loss)
        t.set_description("Loss %3.2e" % loss)

    plt.figure()
    plt.plot(loss)
    plt.savefig("loss-hist.pdf")

    # Do some sampling with the trained results
    
    # Grabbed the trained parameters
    pstore = pyro.get_param_store()
    trained_locs = [pstore[n + "_loc_param"] for n in names]
    trained_scales = [pstore[n + "_scale_param"] for n in names]
    trained_eps = pstore["eps_param"]
    
    # Setup normal distributions
    trained_dist = [torch.distributions.Normal(l, s) for l,s in 
                    zip(trained_locs, trained_scales)]

    # Setup the model
    nsamples = 20
    params = [d.sample((nsamples*ntests,)) for d in trained_dist]
    bmodel = maker(*params)
    # Setup the input
    def expand(d, n):
        return d.unsqueeze(-1).expand(d.shape + (n,)).flatten(-2)

    exp_data = expand(data, nsamples)
    exp_cycles = expand(cycles, nsamples)
    exp_types = expand(types, nsamples)
    exp_control = expand(control, nsamples)
   
    # Run the forward model
    with torch.no_grad():
        exp_results = bmodel.solve_strain(exp_data[0], exp_data[2], exp_data[1])

    exp_stress = exp_results[:,:,0].reshape((exp_results.shape[0],ntests,nsamples))
    mean_stress = torch.mean(exp_stress, dim = -1)

    p = 0.05

    for i in range(0,ntests,100):
        plt.figure()
        plt.plot(data[2,:,i].cpu().numpy(), mean_stress[:,i].cpu().numpy(), 'k-')
        plt.plot(data[2,:,i].cpu().numpy(), results[:,i].cpu().numpy(), 'k--')

        n = exp_stress[:,i].shape[-1]
        l = int(n * p) + 1
        u = int(n * (1-p)) + 1

        lb, _ = torch.kthvalue(exp_stress[:,i], l, dim = -1)
        ub, _ = torch.kthvalue(exp_stress[:,i], u, dim = -1)

        plt.fill_between(data[2,:,i].cpu().numpy(), lb.cpu().numpy(), ub.cpu().numpy(), alpha = 0.2)

        plt.savefig("exp-%i.pdf" % i)
        plt.close()