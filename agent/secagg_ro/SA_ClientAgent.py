from agent.Agent import Agent
from agent.secagg_ro.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util.util import log_print

# from util.crypto.logReg import getWeights, reportStats
import util.crypto.diffieHellman as dh
from util.crypto.FiniteField import FiniteField
# import util.crypto.shamir as shamir
from util.crypto.shamir_gmp import Shamir

import nacl.bindings as nb
import nacl.hash as nh
import nacl.secret
from nacl.encoding import Base64Encoder
from nacl.signing import SigningKey, VerifyKey
import nacl.utils
import numpy as np
from os.path import exists
import pandas as pd
import random
import secrets
from pympler import asizeof
# import cryptography as crypt
# AES-GCM encryption between clients
# from cryptography.hazmat.primitives.ciphers.aead import AESGCM
# Secret sharing for each client and the server
# import Crypto.Protocol.SecretSharing as shamir
# import Crypto.Cipher as AES


# The PPFL_TemplateClientAgent class inherits from the base Agent class.  It has the
# structure of a secure federated learning protocol with secure multiparty communication,
# but without any particular learning or noise methods.  That is, this is a template in
# which the client parties simply pass around arbitrary data.  Sections that would need
# to be completed are clearly marked.

class SA_ClientAgent(Agent):

  def __init__(self, id, name, type,
               peer_list=None,
               iterations=4,
               max_input = 10000,
               key_length = 256,
               prime = -1,
               crypto_utils = None,
               pki_pub = None, pki_priv = None,
               peer_pki_pubkeys = None,
               num_clients = None,
               num_subgraphs = None,
               random_state=None):

    # Base class init.
    super().__init__(id, name, type, random_state)


    # Store the client's peer list (subgraph, neighborhood) with which it should communicate.
    self.peer_list = peer_list
    # self.peer_list.append(self.id)

    # Initialize a tracking attribute for the initial peer exchange and record the subgraph size.
    self.peer_exchange_complete = False
    self.num_peers = len(self.peer_list)
    self.threshold = int(num_clients * 2 / 3) + 1
    self.peer_pki_pubkeys = peer_pki_pubkeys;
    self.pki_pub = pki_pub
    self.pki_priv = pki_priv

    # Record the total number of clients participating in the protocol and the number of subgraphs.
    # Neither of these are part of the protocol, or necessary for real-world implementation, but do
    # allow for convenient logging of progress and results in simulation.
    self.num_clients = num_clients
    self.num_subgraphs = num_subgraphs

    # Record the number of iterations the clients will perform.
    self.no_of_iterations = iterations

    # Initialize a dictionary to remember which peers we have heard from during peer exchange.
    self.peer_received = {}

    # Initialize a dictionary to accumulate this client's timing information by task.
    self.elapsed_time = { 'OFFLINE' : pd.Timedelta(0),
                          'ENCRYPTION_R1' : pd.Timedelta(0),
                          'ENCRYPTION_R2' : pd.Timedelta(0) }
    self.message_size = { 'PUBKEY' : 0,
                          'SHARES' : 0,
                          'MASKED_INPUT' : 0,
                          'R_SHARE_SUM' : 0 }


    # Pre-generate this client's local input for each iteration
    # (for the sake of simulation speed).
    self.input = []
    # This is a faster PRNG than the default,
    # for times when we must select a large quantity of randomness.
    self.prng_input = np.random.Generator(np.random.SFC64())
    # For each iteration, pre-generate a random integer as its secret input;
    for i in range(iterations):
        self.input.append(self.prng_input.integers(low = 0, high = max_input));

    ### Data randomly selected from total training set each iteration, simulating online behavior.
    # for i in range(iterations):
    #     self.input.append(self.prng.integer(input_range));
      # slice = self.prng.choice(range(self.X_train.shape[0]), size = split_size, replace = False)
      #
      # # Pull together the current local training set.
      # self.trainX.append(self.X_train[slice].copy())
      # self.trainY.append(self.y_train[slice].copy())


    # Create dictionaries to hold the public and secure keys for this client,
    self.key_length = key_length // 8
    self.ek_pub = None
    self.ek_priv = None
    # Dictionaries of the symmetric encryption keys
    self.peer_ek = {}
    self.peer_secret_box = {}

    # if prime == -1:
    #   self.finite_field_p = FiniteField()
    #   self.finite_field_q = FiniteField((self.finite_field_p.getPrime() - 1) // 2)
    # else:
    #   self.finite_field_p = FiniteField(prime)
    #   self.finite_field_q = FiniteField((prime - 1) // 2)

    # Record the self mask
    self.self_r = 0
    # Dictionary of the other users' shares
    self.peer_r_share = {}
    # self.ss_sub = Shamir(self.finite_field_q)
    # self.ss_sub.initCoeff(peer_list)
    
    self.finite_field_p = crypto_utils[0]
    self.finite_field_q = crypto_utils[1]
    self.ss_sub = crypto_utils[3]
    self.ro = crypto_utils[4]




    # Create dictionaries to hold the shared key for each peer each iteration and the seed for the
    # following iteration.
    # *** NOT used in the RO protocol
    self.peer_mutual_masks = {}



    ### ADD DIFFERENTIAL PRIVACY CONSTANTS AND CONFIGURATION HERE, IF NEEDED.
    #
    #


    # Iteration counter.
    self.current_iteration = 0
    self.current_base = 0
    # State flag
    self.setup_complete = False




  ### Simulation lifecycle messages.

  def kernelStarting(self, startTime):

    # Initialize custom state properties into which we will later accumulate results.
    # To avoid redundancy, we allow only the first client to handle initialization.
    if self.id == 1:
        self.kernel.custom_state['offline'] = pd.Timedelta(0)
        self.kernel.custom_state['encryption_r1'] = pd.Timedelta(0)
        self.kernel.custom_state['encryption_r2'] = pd.Timedelta(0)

        self.kernel.custom_state['bdw_pubkey'] = 0
        self.kernel.custom_state['bdw_shares'] = 0
        self.kernel.custom_state['bdw_masked_input'] = 0
        self.kernel.custom_state['bdw_r_share_sum'] = 0

    # Find the PPFL service agent, so messages can be directed there.
    self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

    # Request a wake-up call as in the base Agent.  Noise is kept small because
    # the overall protocol duration is so short right now.  (up to one microsecond)
    super().kernelStarting(startTime + pd.Timedelta(self.random_state.randint(low = 0, high = 1000), unit='ns'))


  def kernelStopping(self):

    # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
    # Note that times which should be reported in the mean per iteration are already so computed.
    # These will be output to the config (experiment) file at the end of the simulation.

    self.kernel.custom_state['offline'] += self.elapsed_time['OFFLINE']
    self.kernel.custom_state['encryption_r1'] += (self.elapsed_time['ENCRYPTION_R1'] / self.no_of_iterations)
    self.kernel.custom_state['encryption_r2'] += (self.elapsed_time['ENCRYPTION_R2'] / self.no_of_iterations)

    self.kernel.custom_state['bdw_pubkey'] += self.message_size['PUBKEY']
    self.kernel.custom_state['bdw_shares'] += self.message_size['SHARES']
    self.kernel.custom_state['bdw_masked_input'] += self.message_size['MASKED_INPUT']
    self.kernel.custom_state['bdw_r_share_sum'] += self.message_size['R_SHARE_SUM']

    super().kernelStopping()


  ### Simulation participation messages.

  def wakeup (self, currentTime):
    super().wakeup(currentTime)
    # print("client", self.id,
    #       "wakeup in iteration", self.current_iteration)

    # Record start of wakeup for real-time computation delay..
    dt_wake_start = pd.Timestamp('now')

    ##############################################################
    # Check if the clients are still performing the setup phase. #
    ##############################################################

    # Setup phase
    if self.current_iteration == 0:
      # If DH keys have not been generated yet
      if not self.setup_complete:
        dt_protocol_start = pd.Timestamp('now')

        # Generate DH keys.
        self.ek_pub, self.ek_priv = nb.crypto_kx_keypair()
        # self.s_priv, self.s_pub = nb.crypto_kx_keypair()

        # Sign the public key with the signing key
        # TODO: Include PKI checking. Using plaintext for now.
        # signature = self.pki_priv.sign(self.ek_pub, encoder=Base64Encoder)

        # Elapsed time must be accumulated before sending messages.
        self.recordTime(dt_protocol_start, 'OFFLINE')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        # Send one message to the server,
        # including id, ek_pub, sig
        # Let the server check the signature and forward to other peers
        self.sendMessage(self.serviceAgentID,
                        Message({ "msg" : "SETUP_PUBKEYS",
                          "iteration": self.current_iteration,
                          "sender" : self.id,
                          "pubkey" : self.ek_pub,
                          # "sig"    : signature,
                          }),
                        tag = "comm_offline")
        self.recordBandwidth(self.ek_pub, 'PUBKEY')






  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Setup phase: Round 2
    # Receiving ek_pub with signatures of other clients from the server
    if msg.body['msg'] == "SETUP_PUBKEYS" and self.current_iteration == 0:

        # Record start of message processing.
        # This is still considered part of the offline setup.
        dt_protocol_start = pd.Timestamp('now')

        # Record all public keys for each peer client
        for peer_id, peer_pubkey in msg.body['pubkeys'].items():

            # Verify the signature of the sender client
            # If invalid, then the server is corrupt
            # Abort.
            # TODO:
            # signature = peer_pubkey[1]
            # pki_pub = VerifyKey(self.peer_pki_pubkeys[peer_id], encoder=Base64Encoder)
            # # signature = signature[:-1] + bytes([int(signature[-1]) ^ 1])
            # pki_pub.verify(signature, encoder=Base64Encoder)

            # if peer_pubkey[1] != 0:
            #     return

            # Key agreement to get the symmetric encryption keys and mutual masks
            # Store the agreed keys
            ek = dh.keyexchange(
                peer_id, self.id, self.ek_pub, self.ek_priv, peer_pubkey)
            self.peer_ek[peer_id] = ek
            # print("symmetric key:", (self.peer_ek[peer_id]))
            byte_key = ek.to_bytes(self.key_length, 'big')
            self.peer_secret_box[peer_id] = nacl.secret.SecretBox(byte_key)

        # Check if the set of users I have done keyexchange with is large enough
        # If not, abort;



        # Generate the self mask
        # prng_r = np.random.Generator(np.random.SFC64())
        # self.self_r = prng_r.integers(low = 0, high = self.finite_field.getPrime())
        self.self_r = secrets.randbelow(self.finite_field_q.getPrime())

        # Secret shares it
        # shares = shamir.secretShare(s = self.self_r,
        #                             t = self.threshold,
        #                             # TODO: Guarantee that the list contains itself
        #                             holders = self.peer_list,
        #                             ff = self.finite_field_q)
        shares = self.ss_sub.secretShare(s = self.self_r, 
                                    t = self.threshold,
                                    # TODO: Guarantee that the list contains itself
                                    holders = self.peer_list)

        # Encrypt each share with the symmetric key
        share_ciphers = {}
        for peer_id in self.peer_ek:
            byte_share = int(shares[peer_id]).to_bytes(256, "big")
            share_ciphers[peer_id] = self.peer_secret_box[peer_id].encrypt(byte_share)

        # Accumulate into offline setup.
        self.recordTime(dt_protocol_start, 'OFFLINE')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        # Send the encrypted shares to the server
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SETUP_SHARES_CIPHER",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "share_ciphers": share_ciphers
                        }),
                tag = "comm_offline")
        self.recordBandwidth(share_ciphers, 'SHARES')




    # Setup phase: Round 3
    # Receiving encrypted shares from the other clients
    elif msg.body['msg'] == "SETUP_SHARES_CIPHER" and self.current_iteration == 0:
        # Still offline setup.
        dt_protocol_start = pd.Timestamp('now')

        ciphers = msg.body['shares_cipher']
        # print("client", self.id,
        #       "receives", len(ciphers),
        #       "encrypted shares")
        if len(ciphers) < self.threshold:
            # TODO: If receiving less than t shares, abort
            print(f"Client {self.id} receiving less than t shares")
            self.current_iteration = -1

        # Decrypt the shares with corresponding symm key
        for peer_id in ciphers:
            byte_share = self.peer_secret_box[peer_id].decrypt(ciphers[peer_id])
            self.peer_r_share[peer_id] = int.from_bytes(byte_share, "big")

        # Accumulate into offline setup.
        self.recordTime(dt_protocol_start, 'OFFLINE')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['OFFLINE'] += dt_protocol_end - dt_protocol_start

        # Otherwise, enter iteration 1.
        dt_agg_start = pd.Timestamp('now')
        self.current_iteration += 1
        masked_input = self.aggMaskedInput()

        # Accumulate into round 1 encryption.
        self.recordTime(dt_agg_start, 'ENCRYPTION_R1')
        # dt_agg_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R1'] += dt_agg_end - dt_agg_start

        # # Delay the client for both sets of activities performed.
        # self.setComputationDelay(int((dt_agg_end - dt_protocol_start).to_timedelta64()))

        # Messages must be sent after computation delay is set.
        self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "AGG_MASKED_INPUT",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "masked_input": masked_input,
                    }),
            tag = "comm_online_r1")
        self.recordBandwidth(masked_input, 'MASKED_INPUT')


    # Aggregation phase: Round 2
    # Receiving the online set from the server
    elif msg.body['msg'] == "AGG_ONLINE_SET" and self.current_iteration != 0:
        # Part of online setup.
        dt_protocol_start = pd.Timestamp('now')

        online_set = msg.body["online_set"]
        # print("client", self.id, "receives online set",
        #     online_set, "in iteartion", self.current_iteration)
        r_shares_sum = self.aggShareSum(online_set)

        self.recordTime(dt_protocol_start, 'ENCRYPTION_R2')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R2'] += dt_protocol_end - dt_protocol_start
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        self.sendMessage(self.serviceAgentID,
             Message({ "msg" : "AGG_SHARES_SUM",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "shares_sum": r_shares_sum,
                    }),
             tag = "comm_online_r2")
        # print(self.id, "r_share_sum:", r_shares_sum)
        # print("\tsize of r_share_sum:", asizeof.asizeof(r_shares_sum))
        self.recordBandwidth(r_shares_sum, 'R_SHARE_SUM')



    # Aggregation phase: Round 3
    # Receiving the output from the server
    elif msg.body['msg'] == "AGG_OUTPUT" and self.current_iteration != 0:
        dt_protocol_start = pd.Timestamp('now')

        # End of the iteration
        # In practice, the client can do calculations here
        # based on the output of the server
        # In this simulation we just log the information
        output = msg.body['output']

        # print(f"User {self.id} completes iteration {self.current_iteration}")

        # Accumulate to encryption.
        # self.recordTime(dt_protocol_start, 'ENCRYPTION_R1')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R1'] += dt_protocol_end - dt_protocol_start

        # # Set delay in case we do not start a new round.
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        # Reset temp variables for each iteration

        # Enter next iteration
        self.current_iteration += 1

        # Start a new iteration
        if self.current_iteration > self.no_of_iterations:
          # TODO: End the protocol
          print("client", self.id, "input list:", self.input)
          return

        dt_part2_start = pd.Timestamp('now')
        masked_input = self.aggMaskedInput()

        # Accumulate to encryption.
        self.recordTime(dt_part2_start, 'ENCRYPTION_R1')
        # dt_part2_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R1'] += dt_part2_end - dt_part2_start

        # # Update delay with complete elapsed time.
        # self.setComputationDelay(int((dt_part2_end - dt_protocol_start).to_timedelta64()))

        # Messages must be sent after computation delay is set.
        self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "AGG_MASKED_INPUT",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "masked_input": masked_input,
                    }),
            tag = "comm_online_r1")
        self.recordBandwidth(masked_input, 'MASKED_INPUT')


        log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, output)

        if self.id == 1: print (f"Protocol iteration {self.current_iteration} complete.")

        # Start a new iteration if we are not at the end of the protocol.
        # if self.current_iteration < self.no_of_iterations:
            # self.setWakeup(currentTime + pd.Timedelta('1ns'))




  #================= Round logics =================#
  def aggMaskedInput(self):
    # Send H(k)^{x_i + r_i} to the server
    # TODO:
    exponent = self.input[self.current_iteration - 1] + self.self_r
    hash = int.from_bytes(
                nh.sha256(self.current_iteration.to_bytes(32, "big")),
                "big")
    base = self.finite_field_p.convert(hash, 2**256)
    while self.finite_field_p.order(base) * 2 + 1 != self.finite_field_p.getPrime():
        # print(f"ERROR: Hash of iteration {self.current_iteration} does not work")
        # return
        base = base + 1
    self.current_base = base
    masked_input = self.finite_field_p.power(base, exponent)
    # masked_input = self.input[self.current_iteration - 1] + self.self_r

    return masked_input



  def aggShareSum(self, online_set):
    r_shares_sum = 0
    for peer_id in online_set:
      r_shares_sum = self.finite_field_q.add(r_shares_sum, self.peer_r_share[peer_id])
    r_shares_sum = self.finite_field_p.power(self.current_base, r_shares_sum)

    return r_shares_sum



  def recordTime(self, startTime, categoryName):
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time[categoryName] += dt_protocol_end - startTime
      self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))
      
  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)