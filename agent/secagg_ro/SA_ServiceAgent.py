from agent.Agent import Agent
from message.Message import Message
from util.util import log_print
# from util.crypto.logReg import getWeights
from util.crypto.FiniteField import FiniteField
import util.crypto.diffieHellman as dh

from copy import deepcopy
import math
import nacl.bindings as nb
import nacl.hash as nh
import nacl.secret
import nacl.utils
import numpy as np
import pandas as pd
import random
from util.crypto.shamir_gmp import Shamir
from pympler import asizeof
# Secret sharing for each client and the server
# import Crypto.Protocol.SecretSharing as shamir

### NEW MESSAGES: CLIENT_WEIGHTS : weights, SHARED_WEIGHTS : weights

# The PPFL_ServiceAgent class inherits from the base Agent class.  It provides
# the simple shared service necessary for model combination under secure
# federated learning.

class SA_ServiceAgent(Agent):

  def __init__(self, id, name, type,
               random_state=None,
               msg_fwd_delay=1000000,
               round_time = pd.Timedelta("10s"),
               users = {},
               prime = -1,
               crypto_utils = None,
               iterations=4,
               num_clients=10,
               offline_rate = 0.0,
               max_input = 10000):

    # Base class init.
    super().__init__(id, name, type, random_state)

    # Total number of clients and the threshold
    self.num_clients = num_clients
    self.num_online = math.floor(num_clients * (1 - offline_rate)) + 1
    if self.num_online > self.num_clients:
      self.num_online = self.num_clients
    self.threshold = int(num_clients * 2 / 3) + 1
    if self.num_online < self.threshold:
      self.num_online = self.threshold
    self.max_input = max_input
    self.max_sum = max_input * num_clients

    # if prime == -1:
    #   self.finite_field_p = FiniteField()
    #   self.finite_field_q = FiniteField((self.finite_field_p.getPrime() - 1) // 2)
    # else:
    #   self.finite_field_p = FiniteField(prime)
    #   self.finite_field_q = FiniteField((prime - 1) // 2)
    # self.finite_field_p.initialize_kangaroo(self.max_sum)

    # How long does it take us to forward a peer-to-peer client relay message?
    self.msg_fwd_delay = msg_fwd_delay
    self.round_time = round_time

    # Agent accumulation of elapsed times by category of task.
    self.elapsed_time = { 'OFFLINE' : pd.Timedelta(0), 
                          'ENCRYPTION_R1' : pd.Timedelta(0),
                          'ENCRYPTION_R2' : pd.Timedelta(0), 
                          # 'AGGREGATION' : pd.Timedelta(0) 
                          }
    self.message_size = { 'PUBKEY' : 0,
                          'SHARES' : 0,
                          'ONLINE_SET' : 0,
                          'OUTPUT' : 0 }

    # How many iterations of the protocol should be run?
    self.no_of_iterations = iterations



    self.users = users # The list of all users id
    self.online_set = {} # temporary online set (for each iteration)

    # Dictionary recording (keyed by agentID):
    # - users and their public encryption keys
    self.user_ek_pubkeys = {}
    # - The encrypted shares received from users in the setup phase
    self.user_shares_cipher = {}
    
    # self.ss = Shamir(self.finite_field_p)
    # # self.ss_sub = Shamir(self.finite_field_q)
    # self.ss.initCoeff(users)
    # self.ss.initSubCoeff(users)
    # # self.ss_sub.initCoeff(users)
    self.finite_field_p = crypto_utils[0]
    self.finite_field_q = crypto_utils[1]
    self.ss = crypto_utils[2]
    self.ro = crypto_utils[4]


    # - The masked input (in exponent)
    #   received from users in each iteartion
    self.user_finish_setup = []
    self.user_masked_input = {}
    # - The sum of shares (in exponent)
    #   received from users in every iteration
    self.user_shares_sum = {}



    # Track the current iteration and round of the protocol.
    self.current_iteration = 0
    self.current_hash = 0
    self.current_round = 0
    # The flag indicating if the server is still waiting for messages
    #   (Used in wakeup call)
    # self.waiting = false

    # Mapping the message processing functions
    self.setupProcessingMap = {
          0: self.setupFirstRound,
          1: self.setupKeyExchange,
          2: self.setupShareCipher,
        }
    self.aggProcessingMap = {
          1: self.aggMaskedInput,
          2: self.aggShareSum,
        }


  ### Simulation lifecycle messages.
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()

    # Initialize custom state properties into which we will accumulate results later.
    self.kernel.custom_state['srv_offline'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_encryption_r1'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_encryption_r2'] = pd.Timedelta(0)
    # self.kernel.custom_state['srv_aggregation'] = pd.Timedelta(0)
    
    self.kernel.custom_state['srv_bdw_pubkey'] = 0
    self.kernel.custom_state['srv_bdw_shares'] = 0
    self.kernel.custom_state['srv_bdw_online_set'] = 0
    self.kernel.custom_state['srv_bdw_output'] = 0

    # This agent should have negligible (or no) computation delay until otherwise specified.
    self.setComputationDelay(0)

    # Request a wake-up call as in the base Agent.
    super().kernelStarting(startTime)


  def kernelStopping(self):
    # Add the server time components to the custom state in the Kernel, for output to the config.
    # Note that times which should be reported in the mean per iteration are already so computed.
    self.kernel.custom_state['srv_offline'] += self.elapsed_time['OFFLINE']
    self.kernel.custom_state['srv_encryption_r1'] += (self.elapsed_time['ENCRYPTION_R1'] / self.no_of_iterations)
    self.kernel.custom_state['srv_encryption_r2'] += (self.elapsed_time['ENCRYPTION_R2'] / self.no_of_iterations)
    # self.kernel.custom_state['srv_aggregation'] += (self.elapsed_time['AGGREGATION'] / self.no_of_iterations)


    self.kernel.custom_state['srv_bdw_pubkey'] += self.message_size['PUBKEY']
    self.kernel.custom_state['srv_bdw_shares'] += self.message_size['SHARES']
    self.kernel.custom_state['srv_bdw_online_set'] += self.message_size['ONLINE_SET']
    self.kernel.custom_state['srv_bdw_output'] += self.message_size['OUTPUT']

    # Allow the base class to perform stopping activities.
    super().kernelStopping()


  ### Simulation participation messages.

  # The service agent wakeup at the end of each round
  # More specifically, it stores the messages on receiving the msgs;
  # When the timing out happens, or it collects enough number of msgs,
  # (i.e., from all clients it is waiting for),
  # it starts processing and replying the messages.
  def wakeup(self, currentTime):
    super().wakeup(currentTime)
    # print("server", self.id, "wakeup in iteration",
    #         self.current_iteration,
    #         "round", self.current_round)
    # if not self.waiting:
    #   return

    # Still in the setup phase
    if self.current_iteration == 0:
      self.setupProcessingMap[self.current_round](currentTime)


    # In the k-th iteration
    else:
      self.aggProcessingMap[self.current_round](currentTime)

  # On receiving messages
  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    # Get the sender's id
    sender_id = msg.body['sender']
    if msg.body['iteration'] != self.current_iteration:
      print("ERROR: server: expecting message from iteration",
            self.current_iteration,
            "but receives memssage of iteration",
            msg.body['iteration'])
      return


    # SETUP phase:
    # Round 1: receiving ek_pub and s_pub from each client
    if msg.body['msg'] == "SETUP_PUBKEYS":
      # Verify the signature,
      # if the signature is not valid, ignore the message

      dt_protocol_start = pd.Timestamp('now')

      # Store the encryption pubkeys
      # self.user_ek_pubkeys[sender_id] = (msg.body['pubkey'], msg.body['sig'])
      self.user_ek_pubkeys[sender_id] = msg.body['pubkey']
      # print("server receives public encryption key",
      #       # msg.body['pubkey'],
      #       "from client",
      #       msg.body['sender'])

      # Accumulate into offline setup.
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
      self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))


    # Round 2: Receiving encrypted shares from a user i
    elif msg.body['msg'] == "SETUP_SHARES_CIPHER":
      dt_protocol_start = pd.Timestamp('now')

      ciphers = msg.body['share_ciphers']
      # If the share for any online user is missing, ignore the user
      for id in self.user_ek_pubkeys:
        if id not in ciphers:
          return

      # print("server receives encrypted shares",
      #       msg.body['share_ciphers'],
      #       "from client",
      #       msg.body['sender'])

      # Otherwise, update the ciphertext dict for each user j
      self.user_finish_setup.append(sender_id)
      for id in self.user_ek_pubkeys:
        self.user_shares_cipher[id][sender_id] = ciphers[id]

      # Accumulate into offline setup.
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
      self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

    # AGG phase:
    # Round 1: Receiving a masked input from a user i
    elif msg.body['msg'] == "AGG_MASKED_INPUT":
      dt_protocol_start = pd.Timestamp('now')

      # If the user does not finish the setup phase, ignore it
      if sender_id not in self.user_finish_setup:
        return
      # store the masked input
      self.user_masked_input[sender_id] = msg.body['masked_input']

      # print("server receives masked input",
      #       msg.body['masked_input'],
      #       "from client",
      #       msg.body['sender'])

      # Accumulate into elapsed time.
      self.recordTime(dt_protocol_start, 'ENCRYPTION_R1')
      # dt_protocol_end = pd.Timestamp('now')
      # self.elapsed_time['ENCRYPTION_R1'] += dt_protocol_end - dt_protocol_start
      # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))


    # Round 2: Receiving the sums of the shares from the user i
    elif msg.body['msg'] == "AGG_SHARES_SUM":
        dt_protocol_start = pd.Timestamp('now')

        # Store the share of the sum
        self.user_shares_sum[sender_id] = msg.body['shares_sum']

        # print("server receives sum of shares",
        #       msg.body['shares_sum'],
        #       "from client",
        #       msg.body['sender'])

        # Accumulate into elapsed time.
        self.recordTime(dt_protocol_start, 'ENCRYPTION_R2')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R2'] += dt_protocol_end - dt_protocol_start
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))




  ### Processing and replying the messages.
  def setupFirstRound(self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    self.current_round = 1
    self.setWakeup(currentTime + self.round_time)

    # Accumulate into offline setup.
    self.recordTime(dt_protocol_start, 'OFFLINE')
    # dt_protocol_end = pd.Timestamp('now')
    # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
    # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))



  def setupKeyExchange (self, currentTime):
    # dt_protocol_start = pd.Timestamp('now')

    # print("server receives ",
    #     len(self.user_ek_pubkeys),
    #     "public keys in iteration",
    #     self.current_iteration)

    # Accumulate into offline setup.
    # dt_protocol_end = pd.Timestamp('now')
    # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
    # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

    # Check if receiving enough public keys at the end of the round
    if len(self.user_ek_pubkeys) < self.threshold:
      # TODO: Abort.
      print(f"ERROR-Server: Not enough number of public keys received (expected:{self.threshold}, receive:{len(self.user_ek_pubkeys)})")
      self.current_iteration = -1
    else:
      for id in self.user_ek_pubkeys:
        # Each online user should receive others' public keys
        self.sendMessage(id,
                Message({"msg" : "SETUP_PUBKEYS",
                        "pubkeys" : self.user_ek_pubkeys,
                        }),
                tag = "comm_offline_server")
        self.recordBandwidth(self.user_ek_pubkeys, 'PUBKEY')

        # Each online user is supposed to receive share ciphers
        self.user_shares_cipher[id] = {}
        self.current_round = 2
        self.setWakeup(currentTime + self.round_time)



  def setupShareCipher (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # print("server receives ",
    #     len(self.user_shares_cipher),
    #     "encrypted shares in iteration",
    #     self.current_iteration)
    
    # Check if enough users reply
    if len(self.user_shares_cipher) < self.threshold:
      # TODO: Abort.
      self.current_iteration = -1
      return

    # Accumulate into offline setup.
    cal_time = self.recordTime(dt_protocol_start, 'OFFLINE')
    # dt_protocol_end = pd.Timestamp('now')
    # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
    # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

    # Reorganize the shares
    # THINK: where should I put this reorganization?

    # Send shares to users
    for id in self.user_shares_cipher:
      self.sendMessage(id,
              Message({"msg" : "SETUP_SHARES_CIPHER",
                      "shares_cipher" : self.user_shares_cipher[id],
                      }),
              tag = "comm_offline_server")
      self.recordBandwidth(self.user_shares_cipher[id], 'SHARES')

    self.current_round = 1
    self.current_iteration = 1
    self.setWakeup(currentTime + self.round_time + cal_time)


  def aggMaskedInput (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # Check if the online user set is large enough
    # If not, skip this iteration
    # print("server receives ", len(self.user_masked_input), "masked input in iteration", self.current_iteration)
    if len(self.user_masked_input) < self.threshold:
      # TODO: let the users know - special message
      # self.current_iteration += 1
      # self.setWakeup(currentTime + self.round_time)
      return
    self.online_set = list(self.user_masked_input.keys())
    # print(self.online_set)
    self.online_set = random.sample(self.online_set, self.num_online)

    # Accumulate into elapsed time.
    cal_time = self.recordTime(dt_protocol_start, 'ENCRYPTION_R1')
    # dt_protocol_end = pd.Timestamp('now')
    # self.elapsed_time['ENCRYPTION_R2'] += dt_protocol_end - dt_protocol_start
    # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))


    # the server sends the user list to all the online users
    for id in self.online_set:
        self.sendMessage(id,
            Message({ "msg": "AGG_ONLINE_SET",
                    "sender": 0,
                    # TODO: sender ID?
                    "online_set": self.online_set
                    }),
            tag = "comm_online_server_r1")
        self.recordBandwidth(self.online_set, 'ONLINE_SET')

    self.current_round = 2
    self.setWakeup(currentTime + self.round_time + cal_time)



  def aggShareSum (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # Check if enough number of shares are received
    # If not, skip the iteration
    if len(self.user_shares_sum) < self.threshold:
      self.current_iteration = -1
      return

    # r_sum = shamir.reconstructInExponent(self.user_shares_sum,
    #                             self.threshold,
    #                             self.finite_field_p)
    r_sum = self.ss.reconstructInExponent(self.user_shares_sum, self.threshold)
    # print("server: r_sum reconstructed in iteration", self.current_iteration,
    #     ":", r_sum)

    # TODO: Bruteforce to get the sum of input
    # Current: Minus directly
    masked_sum = 1
    for id in self.online_set:
        masked_sum = self.finite_field_p.multiply(self.user_masked_input[id], masked_sum)
    # for _, v in self.user_masked_input.items():
    #     masked_sum = self.finite_field_p.multiply(v, masked_sum)
    hash = int.from_bytes(
                nacl.hash.sha256(self.current_iteration.to_bytes(32, "big")),
                "big")
    base = base = self.finite_field_p.convert(hash, 2 ** 256)
    while self.finite_field_p.order(base) * 2 + 1 != self.finite_field_p.getPrime():
        base = base + 1
    x_sum = self.finite_field_p.log(self.finite_field_p.divide(masked_sum, r_sum), base)
    print(f"result of iteration {self.current_iteration}: {x_sum}")

    # Accumulate into elapsed time.
    cal_time = self.recordTime(dt_protocol_start, 'ENCRYPTION_R2')
    # dt_protocol_end = pd.Timestamp('now')
    # self.elapsed_time['AGGREGATION'] += dt_protocol_end - dt_protocol_start
    # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

    # Send the result back to each client.
    for id in self.user_finish_setup:
      self.sendMessage(id,
        Message({ "msg": "AGG_OUTPUT",
                "sender": 0,
                "output": x_sum
                }),
        tag = "comm_online_server_r2")
      self.recordBandwidth(x_sum, 'OUTPUT')


    # Reset iteration variables
    self.user_masked_input = {}
    self.user_shares_sum = {}
    self.current_round = 1

    # End of the iteration
    self.current_iteration += 1
    if (self.current_iteration > self.no_of_iterations):
      # TODO: Tell the server to terminate the simulation
      # updateAgentState(self, state)
      return
    self.setWakeup(currentTime + self.round_time + cal_time)




  def aggShareSum_plaintext (self, currentTime):
    # Check if enough number of shares are received
    # If not, skip the iteration

    dt_protocol_start = pd.Timestamp('now')

    # TODO: Do the reconstruction in the exponent
    # Current: reconstruct directly
    r_sum = shamir.reconstruct(self.user_shares_sum,
                                self.threshold,
                                self.finite_field)
    # print("server: r_sum reconstructed in iteration", self.current_iteration,
    #     ":", r_sum)

    # TODO: Bruteforce to get the sum of input
    # Current: Minus directly
    masked_sum = 0
    for _, v in self.user_masked_input.items():
        masked_sum = self.finite_field.add(v, masked_sum)
    x_sum = self.finite_field.subtract(masked_sum, r_sum)
    print(f"result of iteration {self.current_iteration}: {x_sum}")

    # Accumulate into elapsed time.
    dt_protocol_end = pd.Timestamp('now')
    self.elapsed_time['AGGREGATION'] += dt_protocol_end - dt_protocol_start
    self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

    # Send the result back to each client.
    for id in self.user_finish_setup:
      self.sendMessage(id,
        Message({ "msg": "AGG_OUTPUT",
                "sender": 0,
                # TODO: sender ID?
                "output": x_sum
                }),
        tag = "comm_online_server_r2")

    # Reset iteration variables
    self.user_masked_input = {}
    self.user_shares_sum = {}
    self.current_round = 1

    # End of the iteration
    self.current_iteration += 1
    if (self.current_iteration > self.no_of_iterations):
      # TODO: Tell the server to terminate the simulation
      # updateAgentState(self, state)
      return
    self.setWakeup(currentTime + self.round_time)


  def recordTime(self, startTime, categoryName):
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time[categoryName] += dt_protocol_end - startTime
      self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))

      return dt_protocol_end - startTime

  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)