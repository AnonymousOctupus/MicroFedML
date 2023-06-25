from agent.Agent import Agent
from message.Message import Message
from util.util import log_print
# from util.crypto.logReg import getWeights
from util.crypto.FiniteField import FiniteField

import util.crypto.diffieHellman as dh
import util.prg as prg

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
               field_size = -1,
               iterations=4,
               num_clients=10,
               offline_rate=0.0,
               users = {},
               max_input = 10,
               crypto_utils = None):

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

    # self.finite_field = FiniteField(field_size)

    # How long does it take us to forward a peer-to-peer client relay message?
    self.msg_fwd_delay = msg_fwd_delay
    self.round_time = round_time

    # Agent accumulation of elapsed times by category of task.
    self.elapsed_time = { 'KEY_GENERATION' : pd.Timedelta(0),
                          'SECRET_SHARING' : pd.Timedelta(0),
                          'MASKED_INPUT' : pd.Timedelta(0),
                          'AGGREGATION' : pd.Timedelta(0) }
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
    self.user_pubkeys = {}
    # - The encrypted shares and plain shares
    #   received from users in the setup phase
    self.user_shares_cipher = self.initializeSharesCipher()
    self.user_shares = self.initializeSharesCipher()
    # self.ss = Shamir(self.finite_field)
    # self.ss.initCoeff(users)
    self.finite_field = crypto_utils[0]
    self.ss = crypto_utils[2]

    # - The masked input (in exponent)
    #   received from users in each iteartion
    self.user_finish_setup = []
    self.user_masked_input = {}

    # Track the current iteration and round of the protocol.
    self.current_iteration = 1
    self.current_hash = 0
    self.current_round = 0

    # Mapping the message processing functions
    self.aggProcessingMap = {
          0: self.firstRound,
          1: self.keyExchange,
          2: self.shareCipher,
          3: self.maskedInput,
          4: self.sharesReconstruct,
        }


  ### Simulation lifecycle messages.
  def kernelStarting(self, startTime):
    # self.kernel is set in Agent.kernelInitializing()

    # Initialize custom state properties into which we will accumulate results later.
    self.kernel.custom_state['srv_key_generation'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_secret_sharing'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_masked_input'] = pd.Timedelta(0)
    self.kernel.custom_state['srv_aggregation'] = pd.Timedelta(0)

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
    self.kernel.custom_state['srv_key_generation'] += (self.elapsed_time['KEY_GENERATION'] / self.no_of_iterations)
    self.kernel.custom_state['srv_secret_sharing'] += (self.elapsed_time['SECRET_SHARING'] / self.no_of_iterations)
    self.kernel.custom_state['srv_masked_input'] += (self.elapsed_time['MASKED_INPUT'] / self.no_of_iterations)
    self.kernel.custom_state['srv_aggregation'] += (self.elapsed_time['AGGREGATION'] / self.no_of_iterations)

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

    # In the k-th iteration
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


    # Round 1: receiving ek_pub and mk_pub from each client
    if msg.body['msg'] == "PUBKEYS":
      # Verify the signature,
      # if the signature is not valid, ignore the message

      dt_protocol_start = pd.Timestamp('now')

      # Store the ek and mk pubkeys
      self.user_pubkeys[sender_id] = (
                        msg.body['ek_pubkey'], msg.body['mk_pubkey'])

      # Accumulate into offline setup.
      self.recordTime(dt_protocol_start, "KEY_GENERATION")

      # print("server receives public encryption key",
      #       "from client",
      #       msg.body['sender'])


    # Round 2: Receiving encrypted shares from a user i
    elif msg.body['msg'] == "SHARES_CIPHER":
      dt_protocol_start = pd.Timestamp('now')

      ciphers = msg.body['share_ciphers']
      # If the share for any online user is missing, ignore the user
      for id in self.user_pubkeys:
        if id not in ciphers:
          print(f"ERROR: shares from user {sender_id} for user {id} is missing.")
          return

      # Otherwise, update the ciphertext dict for each user j
      for id in self.user_pubkeys:
        self.user_shares_cipher[id][sender_id] = ciphers[id]

      # Accumulate into offline setup.
      self.recordTime(dt_protocol_start, "SECRET_SHARING")


    # Round 3: Receiving a masked input from a user i
    elif msg.body['msg'] == "MASKED_INPUT":
      dt_protocol_start = pd.Timestamp('now')

      # store the masked input
      self.user_masked_input[sender_id] = msg.body['masked_input']

      # Accumulate into elapsed time.
      self.recordTime(dt_protocol_start, "MASKED_INPUT")

      # print("server receives masked input",
      #     msg.body['masked_input'],
      #     "from client",
      #     sender_id)


    # Round 4: Receiving the shares from the user i
    elif msg.body['msg'] == "SHARES":
        dt_protocol_start = pd.Timestamp('now')

        # Store the shares for each j
        shares = msg.body['shares']
        for peer_id in shares:
          self.user_shares[peer_id][sender_id] = shares[peer_id]

        # Accumulate into elapsed time.
        self.recordTime(dt_protocol_start, "AGGREGATION")



  ### Processing and replying the messages.
  def firstRound(self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    self.current_round = 1
    self.setWakeup(currentTime + self.round_time)

    self.recordTime(dt_protocol_start, "KEY_GENERATION")


  def keyExchange (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')


    # Check if receiving enough public keys at the end of the round
    if len(self.user_pubkeys) < self.threshold:
      self.current_iteration = -1
      return
    else:
      # Accumulate into offline setup.
      cal_time = self.recordTime(dt_protocol_start, "KEY_GENERATION")

      for id in self.user_pubkeys:
        # Each online user should receive others' public keys
        self.sendMessage(id,
                Message({"msg" : "PUBKEYS",
                        "pubkeys" : self.user_pubkeys,
                        }),
                tag = "comm_key_generation_server")
        self.recordBandwidth(self.user_pubkeys, 'PUBKEY')
        
        # Each online user is supposed to receive share ciphers
        self.user_shares_cipher[id] = {}
      self.current_round = 2
      self.setWakeup(currentTime + self.round_time + cal_time)



  def shareCipher (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # print("server receives ",
    #     len(self.user_shares_cipher),
    #     "encrypted shares in iteration",
    #     self.current_iteration)

    # Check if enough users reply
    if len(self.user_shares_cipher) < self.threshold:
      self.current_iteration = -1
      return

    # Reorganize the shares
    # THINK: where should I put this reorganization?

    cal_time = self.recordTime(dt_protocol_start, "SECRET_SHARING")
    # Send shares to users
    for id in self.user_shares_cipher:
      self.sendMessage(id,
              Message({"msg" : "SHARES_CIPHER",
                      "shares_cipher" : self.user_shares_cipher[id],
                      }),
              tag = "comm_secret_sharing_server")
      self.recordBandwidth(self.user_shares_cipher[id], 'SHARES')

    # print("num of shares for each user", self.user_shares_cipher[1])
    self.current_round = 3
    # self.current_iteration = 1
    self.setWakeup(currentTime + self.round_time + cal_time)



  def maskedInput (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # Check if the online user set is large enough
    # If not, skip this iteration
    # print("server receives ", len(self.user_masked_input), "masked input in iteration", self.current_iteration)
    if len(self.user_masked_input) < self.threshold:
      # TODO: let the users know - special message
      print(f"ERROR-Server: Not enough masked input are received." \
      f"(iteration {self.current_iteration}, num of masked input: {len(self.user_masked_input)})")
      # self.current_iteration = -1
      return

    self.online_set = set(self.user_masked_input.keys())
    self.online_set = random.sample(self.online_set, self.num_online)

    # Accumulate into elapsed time
    cal_time = self.recordTime(dt_protocol_start, "MASKED_INPUT")

    # the server sends the user list to all the online users
    for id in self.online_set:
        self.sendMessage(id,
            Message({"msg": "ONLINE_SET",
                    "sender": 0,
                    "online_set": self.online_set
                    }),
            tag = "comm_masked_input_server")
        self.recordBandwidth(self.online_set, 'ONLINE_SET')

    self.current_round = 4
    self.setWakeup(currentTime + self.round_time + cal_time)



  def sharesReconstruct (self, currentTime):
    dt_protocol_start = pd.Timestamp('now')

    # Check if enough number of shares are received
    # If not, skip the iteration
    if len(self.user_shares) < self.threshold:
      self.current_iteration = -1
      print(f"ERROR-Server: Not enough shares are received.")
      return

    mask_sum = 0
    for id in self.user_pubkeys:
      if id in self.online_set:
        r_seed = self.ss.reconstruct(self.user_shares[id],
                                    self.threshold)
        mask_sum = self.finite_field.add(
                        mask_sum,
                        prg.prg(high = self.finite_field.getPrime(),
                                seed = r_seed)[0]
                        )
        # mask_sum += prg.prg(seed = r_seed)[0]
        # print(f"r mask of user {id}: {r_seed}")
      else:
        mk_priv = self.ss.reconstruct(self.user_shares[id],
                                    self.threshold)
        mk_priv = int(mk_priv).to_bytes(32, "big")
        h_mask = 0
        for peer_id in self.user_pubkeys:
          mk_ij = dh.keyexchange(peer_id, id,
                          self.user_pubkeys[id][1], mk_priv,
                          self.user_pubkeys[peer_id][1])
          prg_mk_ij = prg.prg(high = self.finite_field.getPrime(),
                                seed = mk_ij)[0]
          if peer_id < id:
            # h_mask += prg.prg(seed = mk_ij)[0]
            h_mask = self.finite_field.add(h_mask, prg_mk_ij)
          elif peer_id > id:
            # h_mask -= prg.prg(seed = mk_ij)[0]
            h_mask = self.finite_field.subtract(h_mask, prg_mk_ij)

        # print(f"server recovers {id}'s mk_priv: {mk_priv}")
        # print(f"server recovers {id}'s h-mask: {h_mask}")
        # mask_sum -= h_mask
        mask_sum = self.finite_field.subtract(mask_sum, h_mask)

    # print(f"mask sum of iteration {self.current_iteration}: {mask_sum}")


    masked_sum = 0
    for id in self.online_set:
      # masked_sum += self.user_masked_input[id]
      masked_sum = self.finite_field.add(masked_sum, self.user_masked_input[id])


    # x_sum = masked_sum - mask_sum
    x_sum = self.finite_field.subtract(masked_sum, mask_sum)

    # Accumulate into elapsed time.
    cal_time = self.recordTime(dt_protocol_start, "AGGREGATION")

    print(f"Server: finish iteration {self.current_iteration}, outputing {x_sum}")

    # Send the result back to each client.
    for id in self.user_pubkeys:
      self.sendMessage(id,
        Message({ "msg": "OUTPUT",
                "sender": 0,
                "output": x_sum,
                }),
        tag = "comm_output_server")
      self.recordBandwidth(x_sum, 'OUTPUT')


    # Reset iteration variables
    self.user_shares_cipher = self.initializeSharesCipher()
    self.user_masked_input = {}
    self.user_shares = self.initializeSharesCipher()
    self.current_round = 1

    # End of the iteration
    self.current_iteration += 1
    if (self.current_iteration > self.no_of_iterations):
      return
    self.setWakeup(currentTime + self.round_time + cal_time)



# ======================== UTIL ========================
  def initializeSharesCipher(self):
    a = {}
    for i in self.users:
      a[i] = {}
    return a



  def recordTime(self, startTime, categoryName):
      # Accumulate into offline setup.
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time[categoryName] += dt_protocol_end - startTime
      self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))

      return dt_protocol_end - startTime

  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)