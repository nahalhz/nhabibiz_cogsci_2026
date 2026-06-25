
######################
#       IMPORTS      #
######################
import argparse
import os
import csv
import json
import numpy as np
import nengo
import pandas as pd
import matplotlib.pyplot as plt
import sspspace
import scipy
from scipy.stats import beta, norm

import nengo
import nengo_dft
from nengo.config import Config
from nengo.ensemble import Ensemble
from nengo.network import Network
from nengo.networks import Thalamus

# Default synaptic strengths for BG connections.
DEFAULT_WT = 1.0  # standard afferent strength to STN
DEFAULT_WP = 0.9  # standard afferent strength to GPi
DEFAULT_DBS_WT = 0.8  # example reduced STN strength under DBS
DEFAULT_DBS_WP = 0.6  # example reduced GPi strength under DBS
DEFAULT_COMPARE_HEALTHY_DA = 1.5  # preferred healthy dopamine for DBS comparison runs
N_TRIALS = 25

class DNF(Network):
    """A dynamic neural fields network, performing action selection on a bundle of salience-weighted action SSPs.

    Parameters
    ----------
    mode : string, optional (Default: 'continuous')
        The mode for the network to operate in, either continuous or discrete to choose between.
    seed: int, optional (Default: None)
        seed for controlling randomness in network initialisation
    neuron_type : nengo obj, optional (Default: nengo.LIFRate())
        The type of neurons used in the output ensemble. (The DNF network always uses Sigmoid neurons).
    ens_neurons : int, optional (Default: 1000)
        The number of neurons in the output ensemble.
    ens_dims : int, optional (Default: 1)
        Dimensionality of the output ensemble. Should match the dimensionality of the action space.
    dec_neurons : int, optional (Default: 400)
        The number of neurons in the dnf. Equates to the number of place cells used to decode the SSP bundle
        into the salience distribution.
    dnf_h: int, optional (Default: -20)
    dnf_global_inhib: int, optional (Default: 10)
    dnf_tau: float, optional (Default: 0.001)
    kernel_excit: int, optional (Default: 10)
    kernel_inhib: int, optional (Default: 10)
    encoders: array-like, optional (Default: nengo.Default)
        A N X M size array of the encoders for the dnf neurons. 
    """
    def __init__(self, seed=None, 
                 neuron_type=nengo.LIFRate(), 
                 ens_neurons=1000, ens_dims=1,
                 dec_neurons=400, 
                 dnf_h=-20, dnf_global_inhib=10, dnf_tau=0.001,
                 kernel_excit=10, kernel_inhib=0,
                 exc_width=5, inh_width=10,
                 encoders=nengo.Default,
                 radius=1.0,
                 ):

        self.seed = seed
        self.neuron_type = neuron_type 
        self.ens_neurons = ens_neurons
        self.ens_dims = ens_dims
        self.dnf_neurons = dec_neurons
        self.dnf_h = dnf_h
        self.dnf_global_inhib = dnf_global_inhib
        self.dnf_tau = dnf_tau
        self.encoders = encoders
        self.kernel_excit = kernel_excit
        self.kernel_inhib = kernel_inhib
        self.exc_width = exc_width
        self.inh_width = inh_width
            
        config = Config(Ensemble)
        config[Ensemble].neuron_type = self.neuron_type
        
        self.model = nengo.Network(seed=self.seed)
        with self.model:
            # self.state = np.zeros(self.ens_dims) 
            with config:

                ## input node
                self.input = nengo.Node(label="input", size_in=self.ens_dims)
                
                ## layers for breaking bundle into saliences
                self.dnf = nengo_dft.DFT(shape=[self.dnf_neurons], h=self.dnf_h, 
                                    global_inh=self.dnf_global_inhib, tau=self.dnf_tau, 
                                   )
                self.dnf.add_kernel(exc=self.kernel_excit, inh=self.kernel_inhib, 
                               exc_width=self.exc_width, inh_width=self.inh_width)
                
                ## connect stimulus to dft
                nengo.Connection(self.input, self.dnf.s, transform=self.encoders, synapse=None)

                ## create output node for interacting with outside networks 
                self.output = nengo.Node(label="output", size_in=self.ens_dims)
                ## connect neuron activities to output and re-encode as SSP bundle 
                nengo.Connection(self.dnf.g.neurons, self.output, transform=self.encoders.T, synapse=0.01)
                
class DNF_BG_DYNDA(Network):
    """An action selection model using the DNF to perform competition resolution. 
    Takes, as input, a bundle of salience-weighted action SSPs and a float value
    representing the amount of dopamine.

    Parameters
    ----------
    neuron_type : nengo obj, optional (Default: nengo.LIFRate())
        The type of neurons used in the output ensemble. (The DNF network always uses Sigmoid neurons).
    dnf_params :  dictionary, optional (DNF net has its own defaults)
        A dictionary of the values for the parameters of the DNF network
    ssp_dim : int, optional (Default: 1)
        Dimensionality of the SSP input. Should be the same as the ens_dims parameter of the DNF
    """

    def __init__(self, seed=None,
                 neuron_type=nengo.LIFRate(),
                 dnf_params=None,
                 encoders=nengo.Default,
                 d1_weight=1.0,
                 d2_weight=1.0,
                 wt=DEFAULT_WT,
                 wp=DEFAULT_WP,
                ):
        self.seed = seed
        self.ssp_dim = dnf_params['ens_dims']
        self.encoders = encoders
        self.wt = wt
        self.wp = wp

        self.gaba = None #0.008
        self.ampa = 0.0 #0.002

        config = Config(Ensemble)
        config[Ensemble].neuron_type = neuron_type

        # self.model = nengo.Network(seed=self.seed)
        # with self.model:
        with config:

            ## create an input node
            self.input = nengo.Node(label="input", size_in=self.ssp_dim)

            ## create an dopamine node
            self.dopamine = nengo.Node(label="dopamine", size_in=1)

            ## ensemble where we'll multiply the input by the weight +/- dopamine
            inp_times_dop = nengo.Ensemble(n_neurons=1000, dimensions=513, neuron_type=nengo.Direct(), radius=1.0)

            ## create a population of D1 neurons, these will be made to fire faster in the presence of dopamine
            striatum_d1 = DNF(**dnf_params,
                     encoders=self.encoders)
            ## create a population of D2 neurons, these will fire less frequently in the presence of dopamine
            striatum_d2 = DNF(**dnf_params,
                     encoders=self.encoders)
            
            ## now we need the indirect pathway. 
            gpe = nengo.Ensemble(n_neurons=1000,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            neuron_type=nengo.LIFRate()
                            )

            stn = nengo.Ensemble(n_neurons=1000,
                            dimensions=self.ssp_dim,
                            radius=1.0,
                            neuron_type=nengo.LIFRate()
                            )

            ## next we'll set up the direct pathway, creating the GPi and connecting the D1 neurons directly to it
            gpi = nengo.Ensemble(n_neurons=1000,
                                dimensions=self.ssp_dim,
                                radius=1.0,
                                neuron_type=nengo.LIFRate()
                                )

            ## finally, we need an output node to collect the result
            self.output = nengo.Node(size_in=self.ssp_dim)

            def product_d1(x):
                return x[:512] * (d1_weight + x[-1])
            
            def product_d2(x):
                return x[:512] * (d2_weight - x[-1])

            ## transform the input
            nengo.Connection(self.input, inp_times_dop[:512], synapse = None)
            nengo.Connection(self.dopamine, inp_times_dop[-1], synapse = None)
            ## connect input to striatum and stn
            nengo.Connection(inp_times_dop, striatum_d1.input, function=product_d1, synapse=None)
            nengo.Connection(inp_times_dop, striatum_d2.input, function=product_d2, synapse=None)
            nengo.Connection(self.input, stn, synapse=None, transform=self.wt)
            ## indirect pathway connections
            nengo.Connection(striatum_d2.output, gpe, transform=-1.0, synapse=self.gaba)
            nengo.Connection(stn, gpe, transform=1.0, synapse=self.ampa)
            nengo.Connection(gpe, gpi, transform=-1.0, synapse=self.gaba)
            nengo.Connection(gpe, stn, transform=-1.0, synapse=self.gaba)
            ## hyperdirect pathway connections
            nengo.Connection(stn, gpi, transform=self.wp, synapse=self.ampa)
            ## direct pathway connection
            nengo.Connection(striatum_d1.output, gpi, transform=-1.0, synapse=self.gaba)
            ## output connection
            nengo.Connection(gpi, self.output, transform=-3.0, synapse=None)
            
dnf_params = {'dnf_h'           : -3.009416816439706,
              'dnf_global_inhib': 8.641108231311897,
              'dnf_tau'         : 0.04706404390267922,
              'kernel_excit'    : 9.421285790349613,
              'kernel_inhib'    : 1.4841093480380807,
              'exc_width'       : 8.534993467227224,
              'inh_width'       : 4.748154014869723,
              'ens_dims'        : 512,
              'dec_neurons'     : 400,
        }


## create SSP encoder and neuron encoders

## set action space domain
domain = np.arange(0,4,0.01).reshape((-1,1))
## create SSP encoder
ssp_encoder = sspspace.RandomSSPSpace(domain_dim=1, ssp_dim=512, 
                                      rng=np.random.RandomState(), 
                                      length_scale=0.5)
## encode domain as SSPs
domain_phis = ssp_encoder.encode(domain)

## Encoders
low = 0
high = 4
width = high - low
places_ = np.arange(low, high, width/400)
encoders = np.asarray(ssp_encoder.encode(places_.reshape(-1,1))).squeeze()

## Generate a unimodal distribution

## generate random values for the beta distribution
seed = 16
np.random.seed(seed)
a = np.random.uniform(1,10,1)
b = np.random.uniform(1,10,1)

## generate beta distribution
beta_Ps = beta.pdf(domain, a,b, scale=domain[-1])
## encode as an SSP bundle
beta_pattern = np.einsum('n,nd->d', beta_Ps.squeeze(), domain_phis) 


# CSV file (keeps your exact header and order)
CSV_PATH = "trial_results.csv"
DBS_CSV_PATH = "dbs_trial.csv"

DBS_CSV_HEADER = [
    "trial_id",
    "iters",
    "thalamus_threshold",
    "scale_DNF",
    "dopamine_h",
    "dopamine_0",
    "dopamine_d",
    "selected_action",
    "selected_action_0",
    "selected_action_d",
    "std_val",
    "std_val_0",
    "std_val_d",
    "std_idx",
    "std_idx_0",
    "std_idx_d",
    "wt",
    "wp",
    "wt_d",
    "wp_d",
]

# ----------------------------- CSV SETUP / TRIAL-ID HANDLING -----------------------------------
CSV_HEADER = [
    "trial_id",
    "iters",
    "thalamus_threshold",
    "scale_DNF",
    "dopamine",
    "dopamine_0",
    "selected_action",
    "selected_action_0",
    "std_val",
    "std_val_0",
    "std_idx",
    "std_idx_0"
]

def ensure_csv(csv_path: str, header=None):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header if header is not None else CSV_HEADER)

def next_trial_id(csv_path: str) -> int:
    """Auto-increment trial_id by reading the last row (defaults to 1)."""
    if not os.path.exists(csv_path):
        return 1
    last_id = 0
    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            if not row:
                continue
            try:
                last_id = int(row[0])
            except Exception:
                pass
    return last_id + 1 if last_id >= 1 else 1

# ----------------------------- CORE RUNNERS -----------------------------------
def run_one_condition(scale_DNF: float, dopamine: float, thalamus_threshold: float,
                      wt_value: float = DEFAULT_WT, wp_value: float = DEFAULT_WP):
    """
    Mirrors your single-condition code block:
    - builds a fresh Nengo model per trial
    - connects inputs exactly as you do
    - returns selected_action list and the std metrics
    """
    d1_weight = 1.0
    d2_weight = 1.0

    selected_action = []

    for i in range(N_TRIALS):
        # Create nengo model
        model = nengo.Network()
        with model:
            # node for passing in the input pattern
            inp_node = nengo.Node(
                nengo.processes.PresentInput([beta_pattern], presentation_time=1.5)
            )
            # node for passing in the tonic dopamine levels
            da_node = nengo.Node(
                nengo.processes.PresentInput([dopamine], presentation_time=0.5)
            )
            # basal ganglia network
            bg = DNF_BG_DYNDA(
                dnf_params=dnf_params,
                encoders=encoders,
                d1_weight=d1_weight,
                d2_weight=d2_weight,
                wt=wt_value,
                wp=wp_value,
            )
            # Thalamus network (threshold is the variable we sweep)
            thal = Thalamus(dimensions=400, threshold=thalamus_threshold, mutual_inhib=1.0)

            # connect the input to the basal ganglia input
            nengo.Connection(inp_node, bg.input, transform=scale_DNF, synapse=None)
            # connect the dopamine to the basal ganglia dopamine
            nengo.Connection(da_node, bg.dopamine, synapse=None)
            # connect the basal ganglia output to the thalamus input
            nengo.Connection(bg.output, thal.input, transform=bg.encoders, synapse=None)

            # probes for collecting data
            p_in = nengo.Probe(inp_node, synapse=None)
            p_out = nengo.Probe(bg.output, synapse=None)
            t_out = nengo.Probe(thal.output, synapse=None)

        # create and run the simulator
        with nengo.Simulator(model) as sim:
            sim.run(1)

        thal_out = sim.data[t_out][-1]  # shape as produced by your Thalamus config
        action_idx = np.argmax(thal_out)
        action = thal_out[action_idx]
        selected_action.append((action_idx, action))

    # Compute your exact stats
    action_vals = [val for idx, val in selected_action]
    std_val = np.std(action_vals)

    action_indices = [idx for idx, val in selected_action]
    std_idx = np.std(action_indices)

    return selected_action, std_val, std_idx

# ----------------------- CSV FILE PATHS + HEADERS  -----------------------

CSV_PATH = "trial_results.csv"
DBS_CSV_PATH = "dbs_trials.csv"

DBS_CSV_HEADER = [
    "trial_id",
    "iters",
    "thalamus_threshold",
    "scale_DNF",
    "dopamine_h",
    "dopamine_0",
    "dopamine_d",
    "selected_action",      # full list of 25 actions (healthy)
    "selected_action_0",    # full list of 25 actions (unhealthy)
    "selected_action_d",    # full list of 25 actions (dbs)
    "std_val",
    "std_val_0",
    "std_val_d",
    "std_idx",
    "std_idx_0",
    "std_idx_d",
    "wt",
    "wp",
    "wt_d",
    "wp_d",
]

def ensure_csv(csv_path: str, header=None):
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            csv.writer(f).writerow(header)

def next_trial_id(csv_path: str) -> int:
    if not os.path.exists(csv_path):
        return 1
    last = 0
    with open(csv_path, "r") as f:
        r = csv.reader(f)
        next(r, None)
        for row in r:
            try: last = int(row[0])
            except: pass
    return last + 1 if last >= 1 else 1

# ----------------------- CORE RUNNER -----------------------
def run_one_condition(scale_DNF: float, dopamine: float, thalamus_threshold: float,
                      wt_value: float = DEFAULT_WT, wp_value: float = DEFAULT_WP):

    selected_action = []

    for i in range(N_TRIALS):
        model = nengo.Network()
        with model:
            inp_node = nengo.Node(
                nengo.processes.PresentInput([beta_pattern], presentation_time=1.5)
            )
            da_node = nengo.Node(
                nengo.processes.PresentInput([dopamine], presentation_time=0.5)
            )

            bg = DNF_BG_DYNDA(
                dnf_params=dnf_params,
                encoders=encoders,
                d1_weight=1.0,
                d2_weight=1.0,
                wt=wt_value,
                wp=wp_value,
            )
            thal = Thalamus(dimensions=400, threshold=thalamus_threshold, mutual_inhib=1.0)

            nengo.Connection(inp_node, bg.input, transform=scale_DNF, synapse=None)
            nengo.Connection(da_node, bg.dopamine, synapse=None)
            nengo.Connection(bg.output, thal.input, transform=bg.encoders, synapse=None)

            t_out = nengo.Probe(thal.output, synapse=None)

        with nengo.Simulator(model) as sim:
            sim.run(1)

        out = sim.data[t_out][-1]
        idx = int(np.argmax(out))
        val = float(out[idx])
        selected_action.append((idx, val))

    action_vals = [v for _, v in selected_action]
    action_idxs = [i for i, _ in selected_action]

    return selected_action, float(np.std(action_vals)), float(np.std(action_idxs))


def run_dbs_trial_and_log(scale_DNF: float, thalamus_threshold: float,
                          dopamine_healthy: float, dopamine_unhealthy: float, dopamine_dbs: float,
                          csv_path: str = DBS_CSV_PATH,
                          healthy_result=None, unhealthy_result=None,
                          wt_baseline: float = DEFAULT_WT, wp_baseline: float = DEFAULT_WP,
                          wt_dbs: float = DEFAULT_DBS_WT, wp_dbs: float = DEFAULT_DBS_WP):

    if healthy_result is None:
        healthy_result = run_one_condition(scale_DNF, dopamine_healthy,
                                           thalamus_threshold, wt_baseline, wp_baseline)
    if unhealthy_result is None:
        unhealthy_result = run_one_condition(scale_DNF, dopamine_unhealthy,
                                             thalamus_threshold, wt_baseline, wp_baseline)

    dbs_result = run_one_condition(scale_DNF, dopamine_dbs,
                                   thalamus_threshold, wt_dbs, wp_dbs)

    # unpack
    healthy_actions, healthy_std_val, healthy_std_idx = healthy_result
    unhealthy_actions, unhealthy_std_val, unhealthy_std_idx = unhealthy_result
    dbs_actions, dbs_std_val, dbs_std_idx = dbs_result

    ensure_csv(csv_path, header=DBS_CSV_HEADER)
    trial_id = next_trial_id(csv_path)

    # save action arrays
    row = [
        trial_id,
        N_TRIALS,
        thalamus_threshold,
        scale_DNF,
        dopamine_healthy,
        dopamine_unhealthy,
        dopamine_dbs,

        json.dumps(healthy_actions),    # full list of tuples
        json.dumps(unhealthy_actions),  
        json.dumps(dbs_actions),      

        healthy_std_val,
        unhealthy_std_val,
        dbs_std_val,
        healthy_std_idx,
        unhealthy_std_idx,
        dbs_std_idx,
        wt_baseline,
        wp_baseline,
        wt_dbs,
        wp_dbs,
    ]

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow(row)

    print(f"Saved DBS comparison trial {trial_id} to {csv_path}")
    return {
        "trial_id": trial_id,
        "healthy": healthy_result,
        "unhealthy": unhealthy_result,
        "dbs": dbs_result,
    }
if __name__ == "__main__":
#   pairs = [(0.5, 0.6), (0.5, 0.7), (0.6, 0.5), ... , (0.9, 0.8)]
    pairs = [(0.5, 0.6)]
    for i in range(8):
        for (wt, wp) in pairs:
            print(f"\nRepeat {i+1}/20")
            run_dbs_trial_and_log(
                scale_DNF=0.9,
                thalamus_threshold=0.4,
                dopamine_healthy=1.2,
                dopamine_unhealthy=0.10,
                dopamine_dbs=0.10,
                csv_path="dbs_trial_new.csv",
                wt_dbs=wt,
                wp_dbs=wp,
            )
