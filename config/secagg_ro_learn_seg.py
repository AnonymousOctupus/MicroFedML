# Our custom modules.
from collections import UserList
from fileinput import filename
from Kernel import Kernel
from agent.secagg_ro_learn_seg.SA_ClientAgent import SA_ClientAgent as ClientAgent
from agent.secagg_ro_learn_seg.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from model.LatencyModel import LatencyModel
from util import util
import util.crypto.toolkit as toolkit
# from util.crypto.FiniteField import FiniteField
# from util.crypto.shamir_gmp import Shamir
# from util.crypto.RandomOracle import RandomOracle
from util.crypto import logReg

# Standard modules.
from datetime import timedelta
from math import floor
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey
import numpy as np
from os.path import exists
import pandas as pd
from sklearn.model_selection import train_test_split
from sys import exit
from time import time

import pickle


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-a', '--clear_learning', action='store_true',
                    help='Learning in the clear (vs SMP protocol)')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-e', '--epsilon', type=float, default=1.0,
                    help='Privacy loss epsilon')
parser.add_argument('-g', '--num_subgraphs', type=int, default=1,
                    help='Number of connected subgraphs into which to place client agents')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('-m', '--max_logreg_iterations', type=int, default=50,
                    help='Number of iterations for client local LogReg'),
parser.add_argument('--max_input', type=int, default=10000,
                    help='Maximum input of each client'),
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('-o', '--collusion', action='store_true',
                    help='Compute collusion analysis (big and slow!)')
parser.add_argument('--offline_rate', type=float, default=0.0,
                    help='The fraction of offline nodes')
parser.add_argument('-p', '--split_size', type=int, default=20,
                    help='Local training size per client per iteration')
parser.add_argument('-r', '--learning_rate', type=float, default=10.0,
                    help='Local learning rate for training on client data')
parser.add_argument('--round_time', type=int, default=10,
                    help='Fixed time the server waits for one round')
parser.add_argument('-s', '--seed', type=int, default=None,
                    help='numpy.random.seed() for simulation')
parser.add_argument('-v', '--verbose', action='store_true',
                    help='Maximum verbosity!')
parser.add_argument('--config_help', action='store_true',
                    help='Print argument options for this config file')

args, remaining_args = parser.parse_known_args()

if args.config_help:
  parser.print_help()
  exit()

# Historical date to simulate.  Required even if not relevant.
historical_date = pd.to_datetime('2014-01-28')

# Requested log directory.
log_dir = args.log_dir
skip_log = args.skip_log

# Random seed specification on the command line.  Default: None (by clock).
# If none, we select one via a specific random method and pass it to seed()
# so we can record it for future use.  (You cannot reasonably obtain the
# automatically generated seed when seed() is called without a parameter.)

# Note that this seed is used to (1) make any random decisions within this
# config file itself and (2) to generate random number seeds for the
# (separate) Random objects given to each agent.  This ensure that when
# the agent population is appended, prior agents will continue to behave
# in the same manner save for influences by the new agents.  (i.e. all prior
# agents still have their own separate PRNG sequence, and it is the same as
# before)

seed = args.seed
if not seed: seed = int(pd.Timestamp.now().timestamp() * 1000000) % (2**32 - 1)
np.random.seed(seed)

# Config parameter that causes util.util.print to suppress most output.
util.silent_mode = not args.verbose

num_clients = args.num_clients
offline_rate = args.offline_rate
round_time = args.round_time
max_input = args.max_input
num_iterations = args.num_iterations

split_size = args.split_size
max_logreg_iterations = args.max_logreg_iterations
epsilon = args.epsilon
learning_rate = args.learning_rate
clear_learning = args.clear_learning
collusion = args.collusion



### How many client agents will there be?   1000 in 125 subgraphs of 8 fits ln(n), for example
num_subgraphs = args.num_subgraphs

print ("Silent mode: {}".format(util.silent_mode))
print ("Configuration seed: {}\n".format(seed))



# Since the simulator often pulls historical data, we use a real-world
# nanosecond timestamp (pandas.Timestamp) for our discrete time "steps",
# which are considered to be nanoseconds.  For other (or abstract) time
# units, one can either configure the Timestamp interval, or simply
# interpret the nanoseconds as something else.

# What is the earliest available time for an agent to act during the
# simulation?
midnight = historical_date
kernelStartTime = midnight

# When should the Kernel shut down?
kernelStopTime = midnight + pd.to_timedelta('17:00:00')

# This will configure the kernel with a default computation delay
# (time penalty) for each agent's wakeup and recvMsg.  An agent
# can change this at any time for itself.  (nanoseconds)

defaultComputationDelay = 1000000000 * 5   # five seconds

# IMPORTANT NOTE CONCERNING AGENT IDS: the id passed to each agent must:
#    1. be unique
#    2. equal its index in the agents list
# This is to avoid having to call an extra getAgentListIndexByID()
# in the kernel every single time an agent must be referenced.


### Configure the Kernel.
kernel = Kernel("Base Kernel", random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32, dtype='uint64')))

### Obtain random state for whatever latency model will be used.
latency_rstate = np.random.RandomState(seed=np.random.randint(low=0,high=2**32, dtype='uint64'))

### Obtain a seed for the train-test split shuffling.
shuffle_seed = np.random.randint(low=0,high=2**32, dtype='uint64')

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
# accy_multiplier = 1000000
# accy_multiplier = 100
accy_multiplier = 1
offset = 0
segInfo = {"segDigitLength": 4,
          "segNumInt": 2, 
          "segNumDec": 8}



### What will be the scale of the shared secret?
secret_scale = 1000000

### For now, non-integer sizes are NOT handled.  Please choose an even divisor.
subgraph_size = int(floor(num_clients / num_subgraphs))

logReg.number_of_parties = num_clients


### LOAD DATA HERE
#
#   The data should be loaded only once (for speed).  Data should usually be
#   shuffled, split into training and test data, and passed to the client
#   parties.
#
#   X_data should be a numpy array with column-wise features and row-wise
#   examples.  y_data should contain the same number of rows (examples)
#   and a single column representing the label.

X_data, y_data = pickle.load(open('data/adult_data.p', 'rb'))
X_data = np.insert(X_data, 0, 1.0, axis=1)

print (X_data.shape,y_data.shape)
print (np.unique(y_data))


# Randomly shuffle and split the data for training and testing.
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)


#
#   Usually this will be passed through a function to shuffle and split
#   the data into the structures expected by the PPFL clients.  For example:
#   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)
#

# For the template, we generate some random numbers that could serve as example data.
# Note that this data is unlikely to be useful for learning.  It just fills the need
# for data of a specific format.
# example_features = 10
# example_rows = 10000
# X_data = np.random.uniform(size=(example_rows,example_features))
# y_data = np.random.uniform(size=(example_rows,1))

# Randomly shuffle and split the data for training and testing.
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION


### Configure a service agent.

agent_count += 1


### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients
user_list = [*range(a, b)]


# prime = -1
# ff_p = FiniteField(prime)
# ff_p.initialize_kangaroo(max_input * num_clients)
# ff_q = FiniteField(ff_p.getPrime() // 2)
# ss = Shamir(ff_p)
# ss.initCoeff(user_list)
# ss.initSubCoeff(user_list)
# ss_sub = Shamir(ff_q)
# ss_sub.initCoeff(user_list)
# ro = RandomOracle(ff_p)
# crypto_utils = [ff_p, ff_q, ss, ss_sub, ro]
prime = -1
crypto_utils = toolkit.ff_ss_ro(prime, user_list, max_input)


agents.extend([ ServiceAgent(
                id = 0, name = "PPFL Service Agent 0",
                type = "ServiceAgent",
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32, dtype='uint64')),
                msg_fwd_delay=0,
                users = user_list,
                iterations = num_iterations,
                round_time = pd.Timedelta(f"{round_time}s"),
                num_clients = num_clients,
                max_input = max_input,
                offline_rate = offline_rate,
                crypto_utils = crypto_utils,
                weight_length = 105,
                segInfo = segInfo)])
agent_types.extend(["ServiceAgent"])

pki_list = {}
pki_pubkey_list = {}
for i in range(a, b):
  signing_key = SigningKey.generate()
  pki_list[i] = signing_key
  pki_pubkey_list[i] = signing_key.verify_key.encode(encoder = Base64Encoder)

client_init_start = time()

# Iterate over all client IDs.
for i in range (a, b):

  # Determine subgraph.
  subgraph = int(floor((i - a) / subgraph_size))

  #print ("Neighborhood for agent {} is {}".format(i, subgraph))

  # Determine agents in subgraph.
  subgraph_start = a + (subgraph * subgraph_size)
  subgraph_end = a + ((subgraph + 1) * subgraph_size)

  neighbors = range(subgraph_start, subgraph_end)

  #print ("Peers for {} are {}".format(i, [x for x in neighbors if x != i]))

  # Peer list is all agents in subgraph except self.
  agents.append(ClientAgent(id = i,
                name = "PPFL Client Agent {}".format(i),
                type = "ClientAgent",
                peer_list = [ x for x in neighbors],
                peer_pki_pubkeys = pki_pubkey_list,
                pki_priv = pki_list[i],
                iterations = num_iterations,
                max_input = max_input,
                num_clients = num_clients, num_subgraphs = num_subgraphs,
                max_logreg_iterations = max_logreg_iterations, epsilon = epsilon, learning_rate = learning_rate,
                multiplier = accy_multiplier,
                segInfo = segInfo,
                offset = offset,
                X_train = X_train, y_train = y_train,
                X_test = X_test, y_test = y_test,
                split_size = split_size, secret_scale = secret_scale,
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32,  dtype='uint64')),
                crypto_utils = crypto_utils
                ))

agent_types.extend([ "ClientAgent" for i in range(a,b) ])
agent_count += num_clients

client_init_end = time()
init_seconds = client_init_end - client_init_start
td_init = timedelta(seconds = init_seconds)
print (f"Client init took {td_init}")


### Configure a latency model for the agents.

# Get a new-style cubic LatencyModel from the networking literature.
pairwise = (len(agent_types),len(agent_types))

model_args = { 'connected'   : True,

               # All in NYC.
               # Only matters for evaluating "real world" protocol duration,
               # not for accuracy, collusion, or reconstruction.
               'min_latency' : np.random.uniform(low = 21000, high = 100000, size = pairwise),
               'jitter'      : 0.3,
               'jitter_clip' : 0.05,
               'jitter_unit' : 5,
             }

latency_model = LatencyModel ( latency_model = 'cubic',
                              random_state = latency_rstate,
                              kwargs = model_args )


# Start the kernel running.
results = kernel.runner(agents = agents,
                        startTime = kernelStartTime,
                        stopTime = kernelStopTime,
                        agentLatencyModel = latency_model,
                        defaultComputationDelay = defaultComputationDelay,
                        skip_log = skip_log,
                        log_dir = log_dir)


# Print parameter summary and elapsed times by category for this experimental trial.
print ()
print (f"Protocol Iterations: {num_iterations}, Clients: {num_clients}, Split Size: {split_size}, " \
       f"Local Iterations {max_logreg_iterations}, Learning Rate: {learning_rate}.")
print (f"Learning in the clear? {clear_learning}, Privacy Epsilon: {epsilon}.")
print ()
print ("Service Agent mean time per iteration (except offline)...")
print (f"    Offline:       {results['srv_offline']}")
print (f"    Encryption R1: {results['srv_encryption_r1'] / num_iterations}")
print (f"    Encryption R2: {results['srv_encryption_r2'] / num_iterations}")
# print (f"    Aggregation:   {results['srv_aggregation'] / num_iterations}")
print ()
print ("Client Agent mean time per iteration (except offline)...")
print (f"    Offline:        {results['offline'] / num_clients}")
print (f"    Encryption R1:  {results['encryption_r1'] / num_clients}")
print (f"    Encryption R2:  {results['encryption_r2'] / num_clients}")
print ()
print ("Communication / latency mean time per client per iteration (except offline)...")
print (f"    Client Offline: {results['comm_offline'] / num_clients}")
print (f"    Client Online:  {(results['comm_online_r1'] + results['comm_online_r1']) / num_clients}")
print (f"    Server Offline: {results['comm_offline_server'] / num_clients}")
print (f"    Server Online:  {(results['comm_online_server_r1'] + results['comm_online_server_r2']) / num_clients}")
print ()
print (f"Slowest agent simulated time: {results['kernel_slowest_agent_finish_time']}")



# # # Write out the timing log to disk.
# file_name = "results/secagg/ro_timing_log.csv"
# if not exists(file_name):
#   with open(file_name, 'a') as results_file:
#     title = "Clients,Max_input,Offline_rate,Iterations,"
#     title += "Offline,Encryption R1,Encryption R2,"
#     title += "Srv Offline,Srv Encryption R1,Srv Encryption R2,"
#     title += "Client Comm Offline,Client Comm Online R1,Client Comm Online R2,"
#     title += "Srv Comm Offline,Srv Comm Online R1,Srv Comm Online R2,"
#     title += "Last Agent Finish,Time to Simulate\n"
#     results_file.write(title)
#     # results_file.write(f"Clients,Peers,Subgraphs,Iterations,Train Rows,Learning Rate,In The Clear?,Local Iterations,Epsilon,Offline,Encryption R1,Encryption R2,Srv Offline,Srv Encryption R1,Srv Encryption R2,Srv Aggregation,Client Comm Offline,Client Comm Online,Srv Comm Offline,Srv Comm Online,Last Agent Finish,Time to Simulate\n")

# with open(file_name, 'a') as results_file:
#   line =  f"{num_clients},{max_input},{offline_rate},{num_iterations},"
#   # line =  f"{num_clients},{subgraph_size-1},{num_subgraphs},{num_iterations},{split_size},{learning_rate},{clear_learning},{max_logreg_iterations},{epsilon},"
#   line += f"{results['offline'].total_seconds() / num_clients},"
#   line += f"{results['encryption_r1'].total_seconds() / num_clients},"
#   line += f"{results['encryption_r2'].total_seconds() / (num_clients * (1 - offline_rate))},"
#   line += f"{results['srv_offline'].total_seconds()},"
#   line += f"{results['srv_encryption_r1'].total_seconds()},"
#   line += f"{results['srv_encryption_r2'].total_seconds()},"
#   # line += f"{results['srv_aggregation'] / num_iterations},"
#   line += f"{results['comm_offline'].total_seconds() / num_clients},"
#   line += f"{results['comm_online_r1'].total_seconds() / num_clients},"
#   line += f"{results['comm_online_r2'].total_seconds() / (num_clients * (1 - offline_rate))},"
#   line += f"{results['comm_offline_server'].total_seconds() / num_clients},"
#   line += f"{results['comm_online_server_r1'].total_seconds() / num_clients},"
#   line += f"{results['comm_online_server_r2'].total_seconds() / (num_clients * (1 - offline_rate))},"
#   line += f"{results['kernel_slowest_agent_finish_time']},{results['kernel_event_queue_elapsed_wallclock']},\n"

#   results_file.write(line)
