from agent.Agent import Agent
from message.Message import Message
import util.formatting.segmenting as segmenting
# from util.crypto.logReg import getWeights
from util.crypto.FiniteField import FiniteField
from util.crypto.FiniteGroup import FiniteGroup
from util.crypto.RandomOracle import RandomOracle
from util.util import log_print

from copy import deepcopy
import math
import nacl.bindings as nb
import nacl.hash as nh
import nacl.secret
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey, VerifyKey
import nacl.utils
import numpy as np
import pandas as pd
import random
from util.crypto.shamir_cl import Shamir
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
               round_time = pd.Timedelta("100s"),
               prime = -1,
               class_group = None,
               shamir = None,
               iterations=4,
               num_clients=10,
               offline_rate=0.0,
               num_groups = 2,
               threshold = -1,
               users = None,
               groups = None,
               user_pki_pubkeys = None,
               max_input = 10000,
               seg_length = 3):

    # Base class init.
    super().__init__(id, name, type, random_state)

    # Total number of clients and the threshold
    self.num_clients = num_clients
    self.num_groups = num_groups
    self.num_online = math.floor(num_clients * (1 - offline_rate)) + 1
    if self.num_online > self.num_clients:
      self.num_online = self.num_clients
    if threshold == -1:
      self.threshold = int(num_clients * 2 / 3) + 1
    else:
      self.threshold = threshold
    # print("online set size",self.num_online, "threshold:", self.threshold)
    if self.num_online < self.threshold * num_groups:
      self.num_online = self.threshold
    print("online set size",self.num_online)
    self.max_input = max_input
    self.max_sum = max_input * num_clients
    self.finite_group = FiniteGroup(self.max_sum)
    self.seg_length = seg_length
    self.seg_num = segmenting.segNum(self.max_sum, seg_length)

    # if prime == -1:
    #   self.finite_field_p = FiniteField()
    #   self.finite_field_q = FiniteField((self.finite_field_p.getPrime() - 1) // 2)
    # else:
    #   self.finite_field_p = FiniteField(prime)
    #   self.finite_field_q = FiniteField((prime - 1) // 2)
    # self.finite_field_p.initialize_kangaroo(self.max_sum)
    # self.ro = RandomOracle(self.finite_field_p)

    # How long does it take us to forward a peer-to-peer client relay message?
    self.msg_fwd_delay = msg_fwd_delay
    self.round_time = round_time

    # Agent accumulation of elapsed times by category of task.
    self.elapsed_time = { 'OFFLINE' : pd.Timedelta(0),
                          'ENCRYPTION_R1' : pd.Timedelta(0),
                          'ENCRYPTION_R2' : pd.Timedelta(0),}
    self.message_size = { 'PUBKEY' : 0,
                          'MK_SHARES' : 0,
                          'RH_SHARES' : 0,
                          'MK_SHARES' : 0,
                          'OFFLINE_SIG' : 0,
                          'OFFLINE_MK_SHARES' : 0,
                          'ONLINE_SET' : 0,
                          'OUTPUT' : 0 }

    # How many iterations of the protocol should be run?
    self.no_of_iterations = iterations



    self.users = users # The list of all users id
    self.groups = groups
    self.online_set = {} # temporary online set (for each iteration)

    # Dictionary recording (keyed by agentID):
    self.user_pki_pubkeys = user_pki_pubkeys
    # - users and their public encryption keys
    self.user_pubkeys = self.initializePubkeyList()
    # self.user_ek_pubkeys = self.initializePubkeyList()
    # self.user_mk_pubkeys_l = self.initializePubkeyList()
    # self.user_mk_pubkeys_r = self.initializePubkeyList()
    # - The encrypted shares received from users in the setup phase
    self.user_mk_shares_cipher_l = {}
    self.user_mk_shares_cipher_r = {}
    self.user_r_shares_cipher = self.initializeSharesCipher()
    self.user_h_shares_cipher = self.initializeSharesCipher()
    self.user_mk_shares_l = {}
    self.user_mk_shares_r = {}
    
    self.cl = class_group
    self.ss = shamir


    self.user_offline_sets = self.initializeOfflineSets()
    # - The masked input (in exponent)
    #   received from users in each iteartion
    self.user_masked_input = self.initializeUserMaskedInput()
    # - The sum of shares (in exponent)
    #   received from users in every iteration
    self.user_shares_sum = self.initializeUserSharesSum()
    # self.user_r_shares_sum = self.initializeUserSharesSum()
    # self.user_h_shares_sum = self.initializeUserSharesSum()


    # The online sets of Setup phase
    self.user_stp_ol_1 = self.initializeOnlineSets()
    self.user_stp_ol_2 = self.initializeOnlineSets()
    self.user_stp_ol_3 = self.initializeOnlineSets()
    self.user_stp_ol_4 = self.initializeOnlineSets()
    self.user_stp_ol_5 = self.initializeOnlineSets()
    # self.user_finish_setup = self.initializeOnlineSets()
    self.user_finish_setup = []


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
          2: self.setupMKPrivShareCipher,
          3: self.setupRHShareCipher,
          4: self.setupOfflineSet,
          5: self.setupMKPrivReconstruction,
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
    # self.kernel.custom_state['srv_store_model'] = pd.Timedelta(0)
    # self.kernel.custom_state['srv_combine_model'] = pd.Timedelta(0)

    self.kernel.custom_state['srv_bdw_pubkey'] = 0
    self.kernel.custom_state['srv_bdw_mk_shares'] = 0
    self.kernel.custom_state['srv_bdw_rh_shares'] = 0
    self.kernel.custom_state['srv_bdw_offline_sig'] = 0
    self.kernel.custom_state['srv_bdw_offline_mk_shares'] = 0
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
    # self.kernel.custom_state['srv_store_model'] += (self.elapsed_time['STORE_MODEL'] / self.no_of_iterations)
    # self.kernel.custom_state['srv_combine_model'] += (self.elapsed_time['COMBINE_MODEL'] / self.no_of_iterations)

    self.kernel.custom_state['srv_bdw_pubkey'] += self.message_size['PUBKEY']
    self.kernel.custom_state['srv_bdw_mk_shares'] += self.message_size['MK_SHARES']
    self.kernel.custom_state['srv_bdw_rh_shares'] += self.message_size['RH_SHARES']
    self.kernel.custom_state['srv_bdw_offline_sig'] += self.message_size['OFFLINE_SIG']
    self.kernel.custom_state['srv_bdw_offline_mk_shares'] += self.message_size['OFFLINE_MK_SHARES']
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

    # Still in the Setup phase
    if self.current_iteration == 0:
      self.setupProcessingMap[self.current_round](currentTime)
    # In the k-th iteration of Aggregation phase
    else:
      self.aggProcessingMap[self.current_round](currentTime)

  # On receiving messages
  def receiveMessage (self, currentTime, msg):
    # Allow the base Agent to do whatever it needs to.
    super().receiveMessage(currentTime, msg)

    # Get the sender's id and group
    sender_id = msg.body['sender']
    sender_group = msg.body['sender_group']
    if msg.body['iteration'] != self.current_iteration:
      print("ERROR: server: expecting message from iteration",
            self.current_iteration,
            "but receives memssage of iteration",
            msg.body['iteration'])
      return


    # SETUP phase:
    # Round 1: receiving ek_pub and mk_pub_l/r from each client
    if msg.body['msg'] == "SETUP_PUBKEYS":
      dt_start = pd.Timestamp('now')

      # Verify the signature,
      # if the signature is not valid, ignore the message
      # signature = msg.body['sig']
      # pki_pub = VerifyKey(self.user_pki_pubkeys[sender_id], encoder=Base64Encoder)
      # pki_pub.verify(signature, encoder=Base64Encoder)

      self.user_stp_ol_1[sender_group].append(sender_id)

      self.recordTime(dt_start, 'OFFLINE')
      # Store the ek_pub in the list
      self.user_pubkeys[sender_group][sender_id]= (
                    msg.body['ek_pub'],
                    msg.body['mk_pub_l'], msg.body['mk_pub_r'])
                    # msg.body['sig'])



    # Round 2: Receiving mk_pub and shares of mk_priv from a user i
    elif msg.body['msg'] == "SETUP_MK_SHARES_CIPHER":
      dt_start = pd.Timestamp('now')
      cipher_l = msg.body['mk_shares_cipher_l']
      cipher_r = msg.body['mk_shares_cipher_r']

      # If there are not enough number of shares
      # or not all shares are for its neighbors,
      # ignore the message
      lgid = self.leftNeighbor(sender_group)
      rgid = self.rightNeighbor(sender_group)
      for id in self.groups[lgid]:
        if id in self.user_stp_ol_1[lgid] and id not in cipher_l:
          return
      for id in self.groups[rgid]:
        if id in self.user_stp_ol_1[rgid] and id not in cipher_r:
          return

      # Store the cipher of mk_priv shares in each receiver's list
      for id in cipher_l:
        self.user_mk_shares_cipher_l[id][sender_id] = cipher_l[id]
      for id in cipher_r:
        self.user_mk_shares_cipher_r[id][sender_id] = cipher_r[id]

      self.user_stp_ol_2[sender_group].append(sender_id)
      self.recordTime(dt_start, 'OFFLINE')



    # Round 3: Receiving encrypted shares of r and h from a user i
    elif msg.body['msg'] == "SETUP_RH_SHARES_CIPHER":
      dt_start = pd.Timestamp('now')

      r_ciphers = msg.body['r_share_ciphers']
      h_ciphers = msg.body['h_share_ciphers']
      # If the share for any online user is missing, ignore the user i
      for id in self.groups[sender_group]:
        if id in self.user_pubkeys and (id not in r_ciphers or id not in h_ciphers):
          return

      # Otherwise, update the ciphertext dict for each user j
      for id in r_ciphers:
        self.user_r_shares_cipher[id][sender_id] = r_ciphers[id]
        self.user_h_shares_cipher[id][sender_id] = h_ciphers[id]

      self.user_stp_ol_3[sender_group].append(sender_id)
      self.recordTime(dt_start, 'OFFLINE')


    # Round 4: Receiving the offline set from a user i
    elif msg.body['msg'] == "SETUP_OFFLINE_SET":
      dt_start = pd.Timestamp('now')

      # TODO: Verify the signature

      self.user_offline_sets[sender_group][sender_id] = msg.body['offline_set']

      self.user_stp_ol_4[sender_group].append(sender_id)
      self.recordTime(dt_start, 'OFFLINE')


    # Round 5: Receiving the mk_priv shares of offline users
    elif msg.body['msg'] == "SETUP_MK_PRIV_SHARES":
      dt_start = pd.Timestamp('now')

      # TODO: Verify the signature

      # Store the shares in each offline user's dic item
      lgid = self.leftNeighbor(sender_group)
      rgid = self.rightNeighbor(sender_group)
      shares_l = msg.body['mk_priv_shares_l']
      shares_r = msg.body['mk_priv_shares_r']
      for id, share in shares_l.items():
        if id not in self.user_mk_shares_l:
          self.user_mk_shares_l[lgid][id] = {}
        self.user_mk_shares_l[lgid][id][sender_id] = share
      for id, share in shares_r.items():
        if id not in self.user_mk_shares_r:
          self.user_mk_shares_r[rgid][id] = {}
        self.user_mk_shares_r[rgid][id][sender_id] = share

      self.user_stp_ol_5[sender_group].append(sender_id)
      self.user_finish_setup.append(sender_id)
      self.recordTime(dt_start, 'OFFLINE')



    # AGG phase:
    # Round 1: Receiving a masked input from a user i
    elif msg.body['msg'] == "AGG_MASKED_INPUT":
      dt_start = pd.Timestamp('now')

      # If the user does not finish the setup phase, ignore it
      if sender_id not in self.user_finish_setup:
        return
      # store the masked input
      self.user_masked_input[sender_id] = msg.body['masked_input']

      self.recordTime(dt_start, 'ENCRYPTION_R1')


    # Round 2: Receiving the sums of the shares from the user i
    elif msg.body['msg'] == "AGG_SHARES_SUM":
      # print("shares sum from user", sender_id)
      dt_start = pd.Timestamp('now')

      # Store the share of the sum
      # for rho in range(self.seg_num):
      share_index = self.groups[sender_group].index(sender_id) + 1
      self.user_shares_sum[sender_group][share_index] = msg.body['shares_sum']
      # self.user_shares_sum[sender_group][sender_id] = msg.body['shares_sum']

      self.recordTime(dt_start, 'ENCRYPTION_R2')


  ### Processing and replying the messages.
  # Before anything happens
  def setupFirstRound(self, currentTime):
    dt_start = pd.Timestamp('now')
    self.current_round = 1
    self.setWakeup(currentTime + self.round_time)

    self.recordTime(dt_start, 'OFFLINE')

  # Setup Phase, round 1
  # Key Exchange for ek
  def setupKeyExchange (self, currentTime):
    dt_start = pd.Timestamp('now')
    # TODO: Check if receiving enough ek_pub for every group



    cal_time = self.recordTime(dt_start, 'OFFLINE')
    # Sending the ek_pub to all online users
    for gid in self.user_pubkeys:
      for id in self.user_pubkeys[gid]:
        lgid = self.leftNeighbor(gid)
        rgid = self.rightNeighbor(gid)
        # print(type(self.user_pubkeys[lgid]), type(self.user_pubkeys[gid]), type(self.user_pubkeys[rgid]))
        # pubkeys = self.user_pubkeys[lgid] | self.user_pubkeys[gid] | self.user_pubkeys[rgid]
        pubkeys = {}
        pubkeys.update(self.user_pubkeys[lgid])
        pubkeys.update(self.user_pubkeys[gid])
        pubkeys.update(self.user_pubkeys[rgid])
        self.sendMessage(id,
            Message({"msg" : "SETUP_PUBKEYS",
                    "pubkeys" : pubkeys,
                    }),
            tag = "comm_offline_server")
        self.recordBandwidth(pubkeys, 'PUBKEY')

        # Each online user is supposed to receive share ciphers
        self.user_mk_shares_cipher_l[id] = {}
        self.user_mk_shares_cipher_r[id] = {}

    self.current_round = 2
    self.setWakeup(currentTime + self.round_time + cal_time)


  # Setup phase, round 2 - Key exchange for mk
  def setupMKPrivShareCipher (self, currentTime):
    dt_start = pd.Timestamp('now')

    # TODO: Check if receiving enough mk_pub for every group

    cal_time = self.recordTime(dt_start, 'OFFLINE')
    # Sending all mk_pub and neighbors' mk_priv shares to every user
    for gid in self.groups:
      for id in self.user_pubkeys[gid]:
        self.sendMessage(id,
              Message({"msg" : "SETUP_MK_SHARES_CIPHER",
                    "mk_shares_cipher_r" : self.user_mk_shares_cipher_l[id],
                    "mk_shares_cipher_l" : self.user_mk_shares_cipher_r[id],
                    }),
              tag = "comm_offline_server")
        self.recordBandwidth(self.user_mk_shares_cipher_l[id], 'MK_SHARES')
        self.recordBandwidth(self.user_mk_shares_cipher_r[id], 'MK_SHARES')

    self.current_round = 3
    self.setWakeup(currentTime + self.round_time + cal_time)


  # Setup phase, round 3
  # Sharing r and h mask
  def setupRHShareCipher (self, currentTime):
    dt_start = pd.Timestamp('now')
    # TODO: Check if receiving enough messages for every group


    cal_time = self.recordTime(dt_start, 'OFFLINE')
    # Send shares of r and h masks to users
    for id in self.user_r_shares_cipher:
      self.sendMessage(id,
             Message({"msg" : "SETUP_RH_SHARES_CIPHER",
                     "r_shares_cipher" : self.user_r_shares_cipher[id],
                     "h_shares_cipher" : self.user_h_shares_cipher[id],
                     }),
            tag = "comm_offline_server")
      self.recordBandwidth(self.user_r_shares_cipher[id], 'RH_SHARES')
      self.recordBandwidth(self.user_h_shares_cipher[id], 'RH_SHARES')
    self.current_round = 4
    self.setWakeup(currentTime + self.round_time + cal_time)



  # Setup phase, round 4
  # Agreeing on offline set
  def setupOfflineSet (self, currentTime):
    dt_start = pd.Timestamp('now')
    # TODO: Check if receiving enough messages for every group


    # Send offline sets with signatures to users in the neighboring groups
    cal_time = self.recordTime(dt_start, 'OFFLINE')
    for gid in self.user_offline_sets:
      lgid = self.leftNeighbor(gid)
      rgid = self.rightNeighbor(gid)
      for id in self.user_offline_sets[gid]:
        self.sendMessage(id,
                Message({"msg" : "SETUP_OFFLINE_SET",
                       # TODO: Currently no signature verification
                        "offline_sets_l" : self.user_offline_sets[lgid],
                        "offline_sets_r" : self.user_offline_sets[rgid],
                        }),
                tag = "comm_offline_server")
        self.recordBandwidth(self.user_offline_sets[lgid], 'OFFLINE_SIG')
        self.recordBandwidth(self.user_offline_sets[rgid], 'OFFLINE_SIG')

    self.current_round = 5
    self.setWakeup(currentTime + self.round_time + cal_time)


  # Setup phase, round 5
  # Reconstruct the mk_priv of offline nodes
  def setupMKPrivReconstruction (self, currentTime):
    dt_start = pd.Timestamp('now')
    # TODO: Check if receiving enough messages for every group


    # Reconstruct and store the mk_priv (plaintext)
    # and calculate the global h mask
    self.h_mask = 0
    for gid, group_shares in self.user_mk_shares_l.items():
      rgid = self.rightNeighbor(sender_group)
      for i, shares in group_shares.items():
        # mk_priv_i = shamir.reconstruct(shares, self.threshold, self.finite_field_q)
        mk_priv_i = self.ss_sub.reconstruct(shares, self.threshold)
        mk_priv_i = int.to_bytes(mk_priv_i, "big")
        mk_pub_i = self.user_mk_pubkeys_r[gid][i]
        h_i = 0
        for id, mk_pub in self.user_mk_pubkeys_l[rgid].items():
          mk_ij = dh.keyexchange(
                        j, i, mk_pub_i, mk_priv_i, mk_pub_j)
          h_i = self.finite_field_q.subtract(h_i, mk_ij)
        self.h_mask = self.finite_field_q.add(self.h_mask, h_i)

    for gid, group_shares in self.user_mk_shares_r.items():
      lgid = self.leftNeighbor(sender_group)
      for i, shares in group_shares.items():
        # mk_priv_i = shamir.reconstruct(shares, self.threshold, self.finite_field_q)
        mk_priv_i = self.ss_sub.reconstruct(shares, self.threshold, self.finite_field_q)
        mk_priv_i = int.to_bytes(mk_priv_i, "big")
        mk_pub_i = self.user_mk_pubkeys_l[gid][i]
        h_i = 0
        for id, mk_pub in self.user_mk_pubkeys_r[lgid].items():
          mk_ij = dh.keyexchange(
                        j, i, mk_pub_i, mk_priv_i, mk_pub_j)
          h_i = self.finite_field_q.add(h_i, mk_ij)
        self.h_mask = self.finite_field_q.add(self.h_mask, h_i)


    # Enter iteration 1.
    cal_time = self.recordTime(dt_start, 'OFFLINE')
    for gid in self.user_pubkeys:
        for id in self.user_pubkeys[gid]:
            # TODO: Should only send to users who finish Setup
            self.sendMessage(id,
                Message({ "msg": "AGG_START",
                        "sender": 0,
                        }),
                tag = "comm_offline_server")
    print("Server finishes Setup phase")

    self.current_round = 1
    self.current_iteration = 1
    self.setWakeup(currentTime + self.round_time + cal_time)


  # Iteration k, round 1
  def aggMaskedInput (self, currentTime):
    dt_start = pd.Timestamp('now')

    self.online_set = list(self.user_masked_input.keys())
    # print(self.online_set)
    self.online_set = random.sample(self.online_set, self.num_online)
    print(self.online_set)


    # Check if enough number of masked inputs are received from each group
    # If not, skip this iteration
    for d in range(self.num_groups):
      if len(set(self.online_set).intersection(self.groups[d])) < self.threshold:
        # TODO: let the users know - special message
        print(f"ERROR[Server] - {self.current_iteration}: not enough ({len(self.online_set)}) masked inputs received from group {d}")
        self.current_iteration = -1
        return


    cal_time = self.recordTime(dt_start, 'ENCRYPTION_R1')
    # the server sends the user list to all the online users
    for id in self.online_set:
        self.sendMessage(id,
            Message({ "msg": "AGG_ONLINE_SET",
                    "sender": 0,
                    "online_set": self.online_set
                    }),
            tag = "comm_online_server_r1")
        self.recordBandwidth(self.online_set, 'ONLINE_SET')

    print(f"Server finishes round 1, iteration {self.current_iteration}")
    self.current_round = 2
    self.setWakeup(currentTime + self.round_time + cal_time)


  # Iteration k, round 2
  def aggShareSum (self, currentTime):
    dt_start = pd.Timestamp('now')

    # Check if enough number of shares are received from each group
    # If not, skip the iteration
    for d in range(self.num_groups):
      if len(self.user_shares_sum[d]) < self.threshold:
        print(f"ERROR[Server] - {self.current_iteration}: not enough ({len(self.user_shares_sum[0][d])}) sum of shares received from group {d}")
        self.current_iteration = -1
        return

    mask_sum = self.cl.power_g(0)
    for d in range(self.num_groups):
      # mask_sum_d = shamir.reconstructInExponent(self.user_shares_sum[d],
      #                         self.threshold,
      #                         self.finite_field_p)
      # print(f"indices: {self.user_shares_sum[d].keys()}")
      mask_sum_d = self.ss.reconstructInExponent(self.user_shares_sum[d],
                              self.threshold, self.cl)
      mask_sum = self.cl.mul(mask_sum, mask_sum_d)


    masked_sum = self.cl.power_g(0)
    for id in self.online_set:
      masked_sum = self.cl.mul(self.user_masked_input[id], masked_sum)
    x_sum = self.cl.log_f(
                    self.cl.div(masked_sum, mask_sum))

    # result = segmenting.combine(x_sum, self.seg_length)
    result = x_sum
    result = result + self.h_mask
    print(f"result of iteration {self.current_iteration}: {result}")

    cal_time = self.recordTime(dt_start, 'ENCRYPTION_R2')
    # Send the result back to each client.
    for id in self.user_finish_setup:
      self.sendMessage(id,
                    Message({ "msg": "AGG_OUTPUT",
                            "sender": 0,
                            "output": result,
                            }),
                    tag = "comm_online_server_r2")
      self.recordBandwidth(result, 'OUTPUT')

    # Reset iteration variables
    self.user_masked_input = self.initializeUserMaskedInput()
    self.user_shares_sum = self.initializeUserSharesSum()
    self.current_round = 1

    # End of the iteration
    self.current_iteration += 1
    if (self.current_iteration > self.no_of_iterations):
      return
    self.setWakeup(currentTime + 10*self.round_time + cal_time)



  # ======================== UTIL ========================
  def initializePubkeyList(self):
    a = {}
    for d in range(self.num_groups):
      a[d] = {}
    return a


  def initializeOnlineSets(self):
    a = {}
    for i in range(self.num_groups):
      a[i] = []
    return a


  def initializeOfflineSets(self):
    a = {}
    for i in range(self.num_groups):
      a[i] = {}
    return a


  def initializeSharesCipher(self):
    a = {}
    for i in self.users:
      a[i] = {}
    return a

  def initializeUserSharesSum(self):
    a = {}
    # for i in range(self.seg_num):
    #   a[i] = {}
    #   for d in range(self.num_groups):
    #     a[i][d] = {}
    for d in range(self.num_groups):
      a[d] = {}
    return a

  def initializeUserMaskedInput(self):
    a = {}
    # for i in range(self.seg_num):
    #   a[i] = {}
    return a

  def initializeSegNum(self):
    x = self.max_input
    num = 0
    while x > 0:
      num = num + 1
      x = x >> self.seg_length
    return num


  def leftNeighbor(self, group_id):
    if group_id == 0:
      return self.num_groups - 1
    return group_id - 1

  def rightNeighbor(self, group_id):
    if group_id == self.num_groups - 1:
      return 0
    return group_id + 1

  def recordTime(self, startTime, categoryName):
    # Accumulate into offline setup.
    dt_protocol_end = pd.Timestamp('now')
    self.elapsed_time[categoryName] += dt_protocol_end - startTime
    self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))

    return dt_protocol_end - startTime

  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)