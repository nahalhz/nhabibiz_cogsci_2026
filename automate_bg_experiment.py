
######################
#       IMPORTS      #
######################
import argparse
import os
import csv
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
                ):
        self.seed = seed
        self.ssp_dim = dnf_params['ens_dims']
        self.encoders = encoders

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
            nengo.Connection(self.input, stn, synapse=None, transform=wt)
            ## indirect pathway connections
            nengo.Connection(striatum_d2.output, gpe, transform=-1.0, synapse=self.gaba)
            nengo.Connection(stn, gpe, transform=1.0, synapse=self.ampa)
            nengo.Connection(gpe, gpi, transform=-1.0, synapse=self.gaba)
            nengo.Connection(gpe, stn, transform=-1.0, synapse=self.gaba)
            ## hyperdirect pathway connections
            nengo.Connection(stn, gpi, transform=wp, synapse=self.ampa)
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


# ----------------------- CSV SETUP / TRIAL-ID HANDLING -----------------------
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

def ensure_csv(csv_path: str):
    """Create CSV with header if it doesn't exist."""
    if not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)

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


# ----------------------------- DBS -----------------------------------

# Set the synaptic strength parameters for BG connections:
# wt: strength of afferents to the STN (standard value is 1.0, can range from 1.0 down to 0)
wt = 1  # set to 0.8 for reduced STN activity (DBS)

# wp: strength of afferents to the GPi (ranges from 0 [no activity] to 0.9 [maximum activity])
wp = 0.9 # set to 0.6 for reduced GPi activity (DBS)



# ----------------------------- SETTING UP PARAMETERS -----------------------------------
# Unhealthy dopamine: 0.08–0.18 (step 0.01)
DOPAMINE_UNHEALTHY_VALUES = np.arange(0.08, 0.13 + 0.001, 0.01)

# Healthy dopamine: 0.2–1.2 (step 0.1)
DOPAMINE_HEALTHY_VALUES = np.arange(1, 1.5 + 0.001, 0.1)

# scale_DNF values: 0.4–0.8 (step 0.02)
SCALE_DNF_VALUES = np.arange(0.7, 0.9 + 0.001, 0.1)

# Thalamus thresholds: 0.2–0.7 (step 0.02)
THALAMUS_THRESHOLDS = np.arange(0.2, 0.5 + 0.001, 0.1)
N_TRIALS = 25

CSV_PATH = "trial_results.csv"

# ----------------------------- CORE RUNNERS -----------------------------------
def run_one_condition(scale_DNF: float, dopamine: float, thalamus_threshold: float):
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
        with nengo.Simulator(model, progress_bar=False) as sim:
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


def run_pair_and_log(scale_DNF: float, thalamus_threshold: float,
                     dopamine_healthy: float, dopamine_unhealthy: float,
                     csv_path: str):
    """
    Runs both your 'healthy' and 'unhealthy' blocks back-to-back,
    then appends a single CSV row using your exact schema and formatting.
    """
    # Healthy
    selected_action, std_val, std_idx = run_one_condition(
        scale_DNF=scale_DNF,
        dopamine=dopamine_healthy,
        thalamus_threshold=thalamus_threshold
    )

    # Unhealthy
    selected_action_0, std_val_0, std_idx_0 = run_one_condition(
        scale_DNF=scale_DNF,
        dopamine=dopamine_unhealthy,
        thalamus_threshold=thalamus_threshold
    )

    # Prepare CSV
    ensure_csv(csv_path)
    trial_id = next_trial_id(csv_path)

    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            trial_id,
            N_TRIALS,
            thalamus_threshold,
            scale_DNF,
            dopamine_healthy,
            dopamine_unhealthy,
            selected_action,      # list of (idx, val) tuples; kept as Python repr in CSV
            selected_action_0,    # same convention for unhealthy
            float(f"{std_val:.12g}"),   # keep numeric; your print was formatted, CSV stores raw
            float(f"{std_val_0:.12g}"),
            float(f"{std_idx:.12g}"),
            float(f"{std_idx_0:.12g}"),
        ])

    print(f"Saved trial {trial_id} to {csv_path}")
    # Also print the stats, mirroring your prints (optional)
    print(f"[healthy]   std(action values): {std_val:.4f} | std(action indices): {std_idx:.4f}")
    print(f"[unhealthy] std(action values): {std_val_0:.4f} | std(action indices): {std_idx_0:.4f}")


# --------------------------------- MAIN --------------------------------------
# ----------------------- CLI -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate BG experiment sweeps or fast single test")
    parser.add_argument("--fast", action="store_true", help="Run only one set of values")
    parser.add_argument("--dopamine_h", type=float, help="Healthy dopamine (fast mode)")
    parser.add_argument("--dopamine_0", type=float, help="Unhealthy dopamine (fast mode)")
    parser.add_argument("--th", type=float, help="Thalamus threshold (fast mode)")
    parser.add_argument("--sc", type=float, help="scale_DNF (fast mode)")

    # NEW: repeat same fast-mode experiment multiple times
    parser.add_argument(
        "--repeat", type=int, default=1,
        help="Number of times to repeat the same fast-mode experiment (default: 1)"
    )

    args = parser.parse_args()

    if args.fast:
        if None in (args.dopamine_h, args.dopamine_0, args.th, args.sc):
            parser.error("--fast requires --dopamine_h --dopamine_0 --th --sc")

        repeats = max(1, int(args.repeat))
        print(
            f"\n[FAST MODE] Running condition (x{repeats}): "
            f"healthy={args.dopamine_h}, unhealthy={args.dopamine_0}, "
            f"threshold={args.th}, scale_DNF={args.sc}"
        )

        for r in range(repeats):
            print(f"\n[FAST MODE] Repeat {r+1}/{repeats}")
            run_pair_and_log(
                scale_DNF=args.sc,
                thalamus_threshold=args.th,
                dopamine_healthy=args.dopamine_h,
                dopamine_unhealthy=args.dopamine_0,
                csv_path=CSV_PATH
            )
    else:
        # Sweep full grid
        for dopamine_h in DOPAMINE_HEALTHY_VALUES:
            for dopamine_0 in DOPAMINE_UNHEALTHY_VALUES:
                for th in THALAMUS_THRESHOLDS:
                    for sc in SCALE_DNF_VALUES:
                        print(f"\n--- Running condition: threshold={th}, scale_DNF={sc}, "
                              f"healthy_dopamine={dopamine_h}, unhealthy_dopamine={dopamine_0}, "
                              f"iters={N_TRIALS} ---")
                        run_pair_and_log(
                            scale_DNF=sc,
                            thalamus_threshold=th,
                            dopamine_healthy=dopamine_h,
                            dopamine_unhealthy=dopamine_0,
                            csv_path=CSV_PATH
                        )
