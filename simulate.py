import numpy as np
from hypergraph import hypergraph
from py.read import read_data
from py.utils import experiment
import pathlib

data = 'contact-high-school'
through_path = 'throughput/' + data

# create the throughput folder for 
pathlib.Path(through_path).mkdir(parents=True, exist_ok=True) 

# subset for convenience purposes -- data subset will start at this timestamp
t_min = 1386330122

# read in the data, represent it as a hypergraph object, and compute the projected dyadic graph (calling it G)
C = read_data(data, prefix = 'data/', t_min = t_min)
C = hypergraph.hypergraph(C)
G = hypergraph.projected_graph(C, as_hyper = True)

# conduct the simulation

# n_steps = round(G.m*np.log(G.m)) # heuristic from coupon-collector problems. 
n_steps = 1000
w, x, w2 = experiment(G, 
                      n_stub = 10**6,          # do stub-labeled MCMC as a warm-start 
                      n_vertex = n_steps/10,   # steps per round
                      n_rounds = 10**4,        # number of sampling rounds
                      sample_after = 10**3,    # give some time to mix
                      message_every = 10**2)   # print a periodic message

# save the results

np.savetxt(through_path + '/w.txt',  w,  "%.4f")
np.savetxt(through_path + '/x.txt',  x,  "%.4f")
np.savetxt(through_path + '/w2.txt', w2, "%.4f")