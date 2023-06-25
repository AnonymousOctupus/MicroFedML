# Our custom modules.
from collections import UserList
from Kernel import Kernel
from agent.google_basic.SA_ClientAgent import SA_ClientAgent as ClientAgent
from agent.google_basic.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from model.LatencyModel import LatencyModel
from util import util
import util.crypto.toolkit as toolkit 

# Standard modules.
from datetime import timedelta
from math import floor
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey
import numpy as np
from os.path import exists
import pandas as pd
# from sklearn.model_selection import train_test_split
from sys import exit
from time import time


# Some config files require additional command line parameters to easily
# control agent or simulation hyperparameters during coarse parallelization.
import argparse

parser = argparse.ArgumentParser(description='Detailed options for PPFL config.')
parser.add_argument('-c', '--config', required=True,
                    help='Name of config file to execute')
parser.add_argument('-i', '--num_iterations', type=int, default=5,
                    help='Number of iterations for the secure multiparty protocol)')
parser.add_argument('-k', '--skip_log', action='store_true',
                    help='Skip writing agent logs to disk')
parser.add_argument('-l', '--log_dir', default=None,
                    help='Log directory name (default: unix timestamp at program start)')
parser.add_argument('--max_input', type=int, default=10000,
                    help='Maximum input of each client'),
parser.add_argument('-n', '--num_clients', type=int, default=5,
                    help='Number of clients for the secure multiparty protocol)')
parser.add_argument('--offline_rate', type=float, default=0.0,
                    help='The fraction of offline nodes')
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
max_input = args.max_input
round_time = args.round_time
num_iterations = args.num_iterations

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
kernelStopTime = midnight + pd.to_timedelta('3000:00:00')

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

### Configure the agents.  When conducting "agent of change" experiments, the
### new agents should be added at the END only.
agent_count = 0
agents = []
agent_types = []

### What accuracy multiplier will be used?
accy_multiplier = 100000

### What will be the scale of the shared secret?
secret_scale = 1000000


# logReg.number_of_parties = num_clients


### LOAD DATA HERE
#
#   The data should be loaded only once (for speed).  Data should usually be
#   shuffled, split into training and test data, and passed to the client
#   parties.
#
#   X_data should be a numpy array with column-wise features and row-wise
#   examples.  y_data should contain the same number of rows (examples)
#   and a single column representing the label.
#
#   Usually this will be passed through a function to shuffle and split
#   the data into the structures expected by the PPFL clients.  For example:
#   X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25, random_state = shuffle_seed)
#

# For the template, we generate some random numbers that could serve as example data.
# Note that this data is unlikely to be useful for learning.  It just fills the need
# for data of a specific format.
example_features = 10
example_rows = 10000
X_data = np.random.uniform(size=(example_rows,example_features))
y_data = np.random.uniform(size=(example_rows,1))

# Randomly shuffle and split the data for training and testing.
# X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.25)

#
#
### END OF LOAD DATA SECTION


agent_types.extend(["ServiceAgent"])
agent_count += 1


### Configure a population of cooperating learning client agents.
a, b = agent_count, agent_count + num_clients
user_list = [*range(a, b)]

prime = -1
crypto_utils = toolkit.ff_ss_ro(prime, user_list, max_input)



pki_list = {}
pki_pubkey_list = {}
for i in range(a, b):
  signing_key = SigningKey.generate()
  pki_list[i] = signing_key
  pki_pubkey_list[i] = signing_key.verify_key.encode(encoder = Base64Encoder)



### Configure a service agent.
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
                crypto_utils = crypto_utils) ])





client_init_start = time()

# Iterate over all client IDs.
for i in range (a, b):

  # Peer list is all agents in subgraph except self.
  agents.append(ClientAgent(id = i,
                name = "PPFL Client Agent {}".format(i),
                type = "ClientAgent",
                peer_list = user_list,
                peer_pki_pubkeys = pki_pubkey_list,
                pki_priv = pki_list[i],
                iterations = num_iterations,
                max_input = max_input,
                num_clients = num_clients, 
                crypto_utils = crypto_utils,
                # multiplier = accy_multiplier, X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                # split_size = split_size, secret_scale = secret_scale,
                random_state = np.random.RandomState(seed=np.random.randint(low=0,high=2**32,  dtype='uint64'))))

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
print (f"Protocol Iterations: {num_iterations}, Clients: {num_clients}, ")
# print (f"Learning in the clear? {clear_learning}, Privacy Epsilon: {epsilon}.")
print ()
print ("Service Agent mean time per iteration (except offline)...")
print (f"    Key generation:       {results['srv_key_generation']}")
print (f"    Secret sharing: {results['srv_secret_sharing'] / num_iterations}")
print (f"    Masked input: {results['srv_masked_input'] / num_iterations}")
print (f"    Aggregation:   {results['srv_aggregation'] / num_iterations}")
print ()
print ("Client Agent mean time per iteration (except offline)...")
print (f"    Key generation: {results['key_generation'] / num_clients}")
print (f"    Secret sharing:  {results['secret_sharing'] / num_clients}")
print (f"    Masked input:  {results['masked_input'] / num_clients}")
print (f"    Sending shares:  {results['sending_shares'] / num_clients}")
print ()
# print ("Communication / latency mean time per client per iteration (except offline)...")
# print (f"    Client Offline: {results['comm_offline'] / num_clients}")
# print (f"    Client Online:  {results['comm_online'] / num_clients}")
# print (f"    Server Offline: {results['comm_offline_server'] / num_clients}")
# print (f"    Server Online:  {results['comm_online_server'] / num_clients}")
# print ()
print (f"Slowest agent simulated time: {results['kernel_slowest_agent_finish_time']}")






# Write out the timing log to disk.
file_name = "results/google/basic_timing_log.csv"
if not exists(file_name):
  with open(file_name, "a") as results_file:
    title = "Clients,Max_input,Offline_rate,Iterations,"
    title += "Client_key_generation,Client_secret_sharing,"
    title += "Client_masked_input,Client_sending_shares,"
    title += "Server_key_generation,Server_secret_sharing,"
    title += "Server_masked_input,Server_aggregation,"
    title += "comm_key_generation,comm_secret_sharing,"
    title += "comm_masked_input,comm_sending_shares,"
    title += "comm_key_generation_server,comm_secret_sharing_server,"
    title += "comm_masked_input_server,comm_output_server,"

    title += "bdw_key_generation,bdw_secret_sharing,"
    title += "bdw_masked_input,bdw_sending_shares,"
    title += "bdw_key_generation_server,bdw_secret_sharing_server,"
    title += "bdw_masked_input_server,bdw_output_server,"

    title += "Last Agent Finish,Time to Simulate\n"
    results_file.write(title)

with open(file_name, 'a') as results_file:
  line =  f"{num_clients},{max_input},{offline_rate},{num_iterations},"
  line += f"{results['key_generation'].total_seconds() / num_clients},"
  line += f"{results['secret_sharing'].total_seconds() / num_clients},"
  line += f"{results['masked_input'].total_seconds() / num_clients},"
  line += f"{results['sending_shares'].total_seconds() / (num_clients * (1 - offline_rate))},"
  line += f"{results['srv_key_generation'].total_seconds()},"
  line += f"{results['srv_secret_sharing'].total_seconds()},"
  line += f"{results['srv_masked_input'].total_seconds()},"
  line += f"{results['srv_aggregation'].total_seconds()},"
  line += f"{results['comm_key_generation'].total_seconds() / num_clients},"
  line += f"{results['comm_secret_sharing'].total_seconds() / num_clients},"
  line += f"{results['comm_masked_input'].total_seconds() / num_clients},"
  line += f"{results['comm_sending_shares'].total_seconds() / (num_clients * (1 - offline_rate))},"
  line += f"{results['comm_key_generation_server'].total_seconds() / num_clients},"
  line += f"{results['comm_secret_sharing_server'].total_seconds() / num_clients},"
  line += f"{results['comm_masked_input_server'].total_seconds() / (num_clients * (1 - offline_rate))},"
  line += f"{results['comm_output_server'].total_seconds() / num_clients},"

  line += f"{results['bdw_pubkey'] / (num_clients * num_iterations)},"
  line += f"{results['bdw_shares_cipher'] / (num_clients * num_iterations)},"
  line += f"{results['bdw_masked_input'] / (num_clients * num_iterations)},"
  line += f"{results['bdw_shares'] / (num_clients * num_iterations)},"
  line += f"{results['srv_bdw_pubkey'] / num_iterations},"
  line += f"{results['srv_bdw_shares'] / num_iterations},"
  line += f"{results['srv_bdw_online_set'] / num_iterations},"
  line += f"{results['srv_bdw_output'] / num_iterations},"

  line += f"{results['kernel_event_queue_elapsed_wallclock']},{results['kernel_slowest_agent_finish_time']}\n"

  results_file.write(line)
