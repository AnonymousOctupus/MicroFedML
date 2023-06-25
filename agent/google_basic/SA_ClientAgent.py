from agent.Agent import Agent
from agent.google_basic.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util.util import log_print

# from util.crypto.logReg import getWeights, reportStats
import util.crypto.diffieHellman as dh
from util.crypto.FiniteField import FiniteField
from util.crypto.shamir_gmp import Shamir
import util.crypto.shamir as shamir
import util.prg as prg

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
               input_range = 10000,
               key_length = 256,
               field_size = -1,
               pki_pub = None, pki_priv = None,
               peer_pki_pubkeys = None,
               num_clients = None,
               max_input = 10000,
               num_subgraphs = None,
               random_state=None,
               crypto_utils = None):

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
    self.elapsed_time = { 'KEY_GENERATION' : pd.Timedelta(0),
                          'SECRET_SHARING' : pd.Timedelta(0),
                          'MASKED_INPUT' : pd.Timedelta(0),
                          'SENDING_SHARES' : pd.Timedelta(0) }
    self.message_size = { 'PUBKEY' : 0,
                          'SHARES_CIPHER' : 0,
                          'MASKED_INPUT' : 0,
                          'SHARES' : 0 }


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
    self.peer_mk = {}
    self.peer_secret_box = {}

    # self.finite_field = FiniteField(field_size)

    # Record the self mask
    self.self_r = 0
    # Dictionary of the other users' shares
    self.peer_r_share = {}
    self.peer_mk_share = {}
    # self.ss = Shamir(self.finite_field)
    # self.ss.initCoeff(peer_list)

    self.finite_field = crypto_utils[0]
    self.ss = crypto_utils[2]




    # Create dictionaries to hold the shared key for each peer each iteration and the seed for the
    # following iteration.
    # *** NOT used in the RO protocol
    self.peer_mutual_masks = {}



    ### ADD DIFFERENTIAL PRIVACY CONSTANTS AND CONFIGURATION HERE, IF NEEDED.
    #
    #


    # Iteration counter.
    self.current_iteration = 1




  ### Simulation lifecycle messages.

  def kernelStarting(self, startTime):

    # Initialize custom state properties into which we will later accumulate results.
    # To avoid redundancy, we allow only the first client to handle initialization.
    if self.id == 1:
        self.kernel.custom_state['key_generation'] = pd.Timedelta(0)
        self.kernel.custom_state['secret_sharing'] = pd.Timedelta(0)
        self.kernel.custom_state['masked_input'] = pd.Timedelta(0)
        self.kernel.custom_state['sending_shares'] = pd.Timedelta(0)

        self.kernel.custom_state['bdw_pubkey'] = 0
        self.kernel.custom_state['bdw_shares_cipher'] = 0
        self.kernel.custom_state['bdw_masked_input'] = 0
        self.kernel.custom_state['bdw_shares'] = 0

    # Find the PPFL service agent, so messages can be directed there.
    self.serviceAgentID = self.kernel.findAgentByType(ServiceAgent)

    # Request a wake-up call as in the base Agent.  Noise is kept small because
    # the overall protocol duration is so short right now.  (up to one microsecond)
    super().kernelStarting(startTime + pd.Timedelta(self.random_state.randint(low = 0, high = 1000), unit='ns'))


  def kernelStopping(self):

    # Accumulate into the Kernel's "custom state" this client's elapsed times per category.
    # Note that times which should be reported in the mean per iteration are already so computed.
    # These will be output to the config (experiment) file at the end of the simulation.

    self.kernel.custom_state['key_generation'] += (self.elapsed_time['KEY_GENERATION'] / self.no_of_iterations)
    self.kernel.custom_state['secret_sharing'] += (self.elapsed_time['SECRET_SHARING'] / self.no_of_iterations)
    self.kernel.custom_state['masked_input'] += (self.elapsed_time['MASKED_INPUT'] / self.no_of_iterations)
    self.kernel.custom_state['sending_shares'] += (self.elapsed_time['SENDING_SHARES'] / self.no_of_iterations)

    self.kernel.custom_state['bdw_pubkey'] += self.message_size['PUBKEY']
    self.kernel.custom_state['bdw_shares_cipher'] += self.message_size['SHARES_CIPHER']
    self.kernel.custom_state['bdw_masked_input'] += self.message_size['MASKED_INPUT']
    self.kernel.custom_state['bdw_shares'] += self.message_size['SHARES']

    super().kernelStopping()


  ### Simulation participation messages.

  def wakeup (self, currentTime):
    super().wakeup(currentTime)
    # print("client", self.id,
    #       "wakeup in iteration", self.current_iteration)
    # Record start of wakeup for real-time computation delay..
    dt_wake_start = pd.Timestamp('now')

    self.generatingNewKeys()



  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)

    # Round 2
    # Receiving ek_pub, mk_pub
    # with signatures of other clients from the server
    if msg.body['msg'] == "PUBKEYS":

        # Record start of message processing.
        # This is still considered part of the offline setup.
        dt_protocol_start = pd.Timestamp('now')

        # Record all public keys for each peer client
        for peer_id, peer_pubkey in msg.body['pubkeys'].items():
            # Key agreement to get the symmetric encryption keys and mutual masks
            # Store the agreed keys
            ek = dh.keyexchange(
                peer_id, self.id, self.ek_pub, self.ek_priv, peer_pubkey[0])
            mk = dh.keyexchange(
                peer_id, self.id, self.mk_pub, self.mk_priv, peer_pubkey[1])
            self.peer_ek[peer_id] = ek
            self.peer_mk[peer_id] = mk

            byte_key = ek.to_bytes(self.key_length, 'big')
            self.peer_secret_box[peer_id] = nacl.secret.SecretBox(byte_key)

        # Check if the set of users I have done keyexchange with is large enough
        # If not, abort;



        # Generate the self mask
        # prng_r = np.random.Generator(np.random.SFC64())
        self.self_r = secrets.randbelow(self.finite_field.getPrime())
        # self.self_r = self.id

        # Secret shares the self mask
        r_shares = self.ss.secretShare(s = self.self_r,
                                    t = self.threshold,
                                    holders = self.peer_list)


        # Secret shares the secret mk key
        mk_shares = self.ss.secretShare(s = int.from_bytes(self.mk_priv, "big"),
                                    t = self.threshold,
                                    holders = self.peer_list)

        # Encrypt each share with the symmetric key
        share_ciphers = {}
        for peer_id in self.peer_ek:
            r_byte_share = int(r_shares[peer_id]).to_bytes(256, "big")
            mk_byte_share = int(mk_shares[peer_id]).to_bytes(256, "big")
            share_ciphers[peer_id] = (self.peer_secret_box[peer_id].encrypt(r_byte_share),
                                    self.peer_secret_box[peer_id].encrypt(mk_byte_share))

        # Accumulate into offline setup.
        self.recordTime(dt_protocol_start, 'SECRET_SHARING')
        # Send the encrypted shares to the server
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SHARES_CIPHER",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "share_ciphers": share_ciphers,
                        }),
                tag = "comm_secret_sharing")
        self.recordBandwidth(share_ciphers, 'SHARES_CIPHER')




    # Round 3
    # Receiving encrypted shares from the other clients
    elif msg.body['msg'] == "SHARES_CIPHER":
        dt_protocol_start = pd.Timestamp('now')

        ciphers = msg.body['shares_cipher']
        # print("client", self.id,
        #       "receives", len(ciphers),
        #       "encrypted shares")
        if len(ciphers) < self.threshold:
            # TODO: If receiving less than t shares, abort
            self.current_iteration = -1
            return

        # Decrypt the shares with corresponding symm key
        for peer_id in ciphers:
            byte_share = self.peer_secret_box[peer_id].decrypt(ciphers[peer_id][0])
            self.peer_r_share[peer_id] = int.from_bytes(byte_share, "big")
            byte_share = self.peer_secret_box[peer_id].decrypt(ciphers[peer_id][1])
            self.peer_mk_share[peer_id] = int.from_bytes(byte_share, "big")

        masked_input = self.maskedInput()

        self.recordTime(dt_protocol_start, 'MASKED_INPUT')

        # Messages must be sent after computation delay is set.
        self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "MASKED_INPUT",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "masked_input": masked_input,
                    }),
            tag = "comm_masked_input")
        self.recordBandwidth(masked_input, 'MASKED_INPUT')




    # Round 4
    # Receiving the online set from the server
    # Sending the shares of r (for online) / mk (for offline) to the server
    elif msg.body['msg'] == "ONLINE_SET":
        dt_protocol_start = pd.Timestamp('now')

        online_set = msg.body["online_set"]
        # print("client", self.id, "receives online set",
        #     online_set, "in iteartion", self.current_iteration)
        shares = {k:v for k, v in self.peer_r_share.items() if k in online_set}
        shares.update({k:v for k, v in self.peer_mk_share.items() if k not in online_set})


        self.recordTime(dt_protocol_start, 'SENDING_SHARES')

        self.sendMessage(self.serviceAgentID,
             Message({ "msg" : "SHARES",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "shares": shares,
                    }),
             tag = "comm_sending_shares")
        self.recordBandwidth(shares, 'SHARES')



    # End of the protocol / start the next iteration
    # Receiving the output from the server
    elif msg.body['msg'] == "OUTPUT":
        dt_protocol_start = pd.Timestamp('now')

        # End of the iteration
        # In practice, the client can do calculations here
        # based on the output of the server
        # In this simulation we just log the information
        output = msg.body['output']

        # print(f"User {self.id} completes iteration {self.current_iteration}")

        # self.recordTime(dt_protocol_start, 'ENCRYPTION_R1')
        # # Accumulate to encryption.
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R1'] += dt_protocol_end - dt_protocol_start
        #
        # # Set delay in case we do not start a new round.
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        # Reset temp variables for each iteration

        # Enter next iteration
        self.current_iteration += 1;

        # Start a new iteration
        if self.current_iteration > self.no_of_iterations:
          # TODO: End the protocol
          print("client", self.id, "input list:", self.input)
          return

        self.generatingNewKeys()

        # log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, output)

        if self.id == 1: print (f"Client 1: Protocol iteration {self.current_iteration} complete.")

        # Start a new iteration if we are not at the end of the protocol.
        # if self.current_iteration < self.no_of_iterations:
            # self.setWakeup(currentTime + pd.Timedelta('1ns'))




  #================= Round logics =================#
  def generatingNewKeys(self):
    ##############################################################
    # Check if the clients are still performing the setup phase. #
    ##############################################################

    dt_protocol_start = pd.Timestamp('now')

    # Generate DH keys.
    self.ek_pub, self.ek_priv = nb.crypto_kx_keypair()
    self.mk_pub, self.mk_priv = nb.crypto_kx_keypair()

    # Elapsed time must be accumulated before sending messages.
    self.recordTime(dt_protocol_start, 'KEY_GENERATION')
    self.sendMessage(self.serviceAgentID,
                  Message({ "msg" : "PUBKEYS",
                    "iteration": self.current_iteration,
                    "sender" : self.id,
                    "ek_pubkey" : self.ek_pub,
                    "mk_pubkey" : self.mk_pub,
                    }),
                  tag = "comm_key_generation")
    self.recordBandwidth(self.ek_pub, 'PUBKEY')
    self.recordBandwidth(self.mk_pub, 'PUBKEY')



  def maskedInput(self):
    masked_input = self.input[self.current_iteration - 1]
    h_mask = 0
    for peer_id in self.peer_mk:
      prg_mk_ij = prg.prg(high = self.finite_field.getPrime(),
                        seed = self.peer_mk[peer_id])[0]
      if peer_id < self.id:
        h_mask = self.finite_field.add(h_mask, prg_mk_ij)
      elif peer_id > self.id:
        h_mask = self.finite_field.subtract(h_mask, prg_mk_ij)
        # h_mask -= prg.prg(seed = self.peer_mk[peer_id])[0]
    # masked_input += h_mask
    masked_input = self.finite_field.add(masked_input, h_mask)

    # print("user", self.id, "'s mk_priv':", self.mk_priv)
    # print("user", self.id, "'s h-mask':", h_mask)
    masked_input = self.finite_field.add(
                        masked_input,
                        prg.prg(high = self.finite_field.getPrime(),
                                seed = self.self_r)[0]
                        )
    # masked_input += prg.prg(seed = self.self_r)[0]
    return masked_input



# ======================== UTIL ========================
  def recordTime(self, startTime, categoryName):
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time[categoryName] += dt_protocol_end - startTime
      self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))

  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)