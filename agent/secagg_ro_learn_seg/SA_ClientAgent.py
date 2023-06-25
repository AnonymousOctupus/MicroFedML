from agent.Agent import Agent
from agent.secagg_ro_learn_seg.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util.formatting.segmenting import segNum
from util.util import log_print

from util.crypto.logReg import getWeights, reportStats
import util.crypto.diffieHellman as dh
from util.crypto.FiniteField import FiniteField
# import util.crypto.shamir as shamir
from util.crypto.shamir_gmp import Shamir

import util.formatting.float_weight_formatting as weightFormat

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
               random_state=None,

              segInfo = None,

               multiplier=10000, secret_scale = 100000,
               offset = 2,
               X_train = None, y_train = None,
               X_test = None, y_test = None,
               split_size = None, learning_rate = None,
               clear_learning = None,
               epsilon = None, max_logreg_iterations = None,
               ):

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

    # Record the multiplier that will be used to protect against floating point accuracy loss and
    # the scale of the client shared secrets.
    self.multiplier = multiplier
    self.offset = offset
    self.secret_scale = secret_scale

    self.segDigitLength = segInfo["segDigitLength"]
    self.segNumInt = segInfo["segNumInt"]
    self.segNumDec = segInfo["segNumDec"]

    # Record the number of local iterations of logistic regression each client will run during
    # each protocol iteration and what local learning rate will be used.
    self.max_logreg_iterations = max_logreg_iterations
    self.learning_rate = learning_rate

    # Record whether clients will do federated learning in the clear (no privacy, no encryption)
    # and, if needed, the epsilon value for differential privacy.
    self.clear_learning = clear_learning
    self.epsilon = epsilon

    # Record the training and testing splits for the data set to be learned.
    self.X_train = X_train
    self.y_train = y_train

    self.X_test = X_test
    self.y_test = y_test

    # Record the number of features in the data set.
    self.no_of_weights = X_train.shape[1]
    # Initialize an attribute to remember the shared weights returned from the server.
    self.prevWeight = None
    # Each client receives only a portion of the training data each protocol iteration.
    self.split_size = split_size

    # Initialize a dictionary to remember which peers we have heard from during peer exchange.
    self.peer_received = {}

    # Initialize a dictionary to accumulate this client's timing information by task.
    self.elapsed_time = { 'OFFLINE' : pd.Timedelta(0),
                          'ENCRYPTION_R1' : pd.Timedelta(0),
                          'ENCRYPTION_R2' : pd.Timedelta(0) }


    # Pre-generate this client's local input for each iteration
    # (for the sake of simulation speed).
    self.trainX = []
    self.trainY = []

    self.input = []
    # This is a faster PRNG than the default,
    # for times when we must select a large quantity of randomness.
    self.prng_input = np.random.Generator(np.random.SFC64())

    # For each iteration, pre-generate a random integer as its secret input; (only secagg scenario)
    for i in range(iterations):
        self.input.append(self.prng_input.integers(low = 0, high = max_input));

    ### Data randomly selected from total training set each iteration, simulating online behavior.
    for i in range(iterations):
      # self.input.append(self.prng.integer(input_range));
      slice = self.prng_input.choice(range(self.X_train.shape[0]), size = split_size, replace = False)

      # Pull together the current local training set.
      self.trainX.append(self.X_train[slice].copy())
      self.trainY.append(self.y_train[slice].copy())


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
    self.peer_mutual_masks = {}


    # Specify the parameters used for generation of randomness.
    self.px_reg = 1
    self.px_epsilon = epsilon
    self.px_min_rows = self.split_size

    self.px_shape = 1 / ( self.num_peers + 1)
    self.px_scale = 2 / (( self.num_peers + 1 ) * self.px_min_rows * self.px_reg * self.px_epsilon )

    if self.id == 1: print (f"px_shape is {self.px_shape}")
    if self.id == 1: print (f"px_scale is {self.px_scale}")

    # Specify the required shape for vectorized generation of randomness.
    self.px_dims = ( self.num_peers, self.no_of_iterations, self.no_of_weights )



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
        # print("client", self.id,
        #       "sends public key", self.ek_pub,
        #       "to the server")






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
        self.self_r = self.id

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



    # Aggregation phase: Round 3
    # Receiving the output from the server
    elif msg.body['msg'] == "AGG_OUTPUT" and self.current_iteration != 0:
        dt_protocol_start = pd.Timestamp('now')

        # End of the iteration
        # In practice, the client can do calculations here
        # based on the output of the server
        # In this simulation we just log the information
        output = msg.body['output']
        # self.prevWeight = msg.body['output']
        self.prevWeight = [0] * self.weight_length
        # print(self.prevWeight)
        for i in range(self.weight_length):
          self.prevWeight[i] = weightFormat.toWeights(output[i][0], output[i][1], 
                                self.segDigitLength, self.segNumInt, self.segNumDec)
          self.prevWeight[i] = self.prevWeight[i] / self.num_clients
        # for i in range(self.weight_length):
        #   self.prevWeight[i] = (float(self.prevWeight[i]) / self.multiplier) - self.offset

        # print(f"User {self.id} completes iteration {self.current_iteration}")

        # Accumulate to encryption.
        # self.recordTime(dt_protocol_start, 'ENCRYPTION_R1')
        # dt_protocol_end = pd.Timestamp('now')
        # self.elapsed_time['ENCRYPTION_R1'] += dt_protocol_end - dt_protocol_start

        # # Set delay in case we do not start a new round.
        # self.setComputationDelay(int((dt_protocol_end - dt_protocol_start).to_timedelta64()))

        # Reset temp variables for each iteration

        # Print current training results
        if self.id == 1:
          #if self.current_iteration in [1, self.no_of_iterations]:
          is_acc, is_mcc, is_f1, is_mse, is_auprc, oos_acc, oos_mcc, oos_f1, oos_mse, oos_auprc = reportStats(self.prevWeight, self.current_iteration, self.X_train, self.y_train, self.X_test, self.y_test)
          # is_acc, is_mcc, is_mse, oos_acc, oos_mcc, oos_mse = reportStats(self.prevWeight, self.current_iteration, self.X_train, self.y_train, self.X_test, self.y_test)
          res_file_name = "results/all_results_with_secagg_ro.csv"
          if not exists(res_file_name):
            with open(res_file_name, 'a') as results_file:
              results_file.write(f"Clients,Iterations,Train Rows,Learning Rate,In The Clear?,Local Iterations,Epsilon,Iteration,IS ACC,OOS ACC,IS MCC,OOS MCC,IS MSE,OOS MSE\n")

          with open(res_file_name, 'a') as results_file:
            results_file.write(f"{self.num_clients},{self.no_of_iterations},{self.split_size},{self.learning_rate},{self.clear_learning},{self.max_logreg_iterations},{self.epsilon},{self.current_iteration},{is_acc},{oos_acc},{is_mcc},{oos_mcc},{is_mse},{oos_mse}\n")


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

        # Messages must be sent after computation delay is set.
        self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "AGG_MASKED_INPUT",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "masked_input": masked_input,
                    }),
            tag = "comm_online_r1")


        log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, self.prevWeight)

        if self.id == 1: print (f"Protocol iteration {self.current_iteration} complete.")





  #================= Round logics =================#
  def aggMaskedInput(self):
    # Perform the local training for this client, using only its local (private) data.  The configured learning
    # rate might need to be increased if there are very many clients, each with very little data, otherwise
    # convergence may take a really long time.
    #
    # max_iter controls how many iterations of gradient descent to perform on the logistic
    # regression model.  previous_weight should be passed as None for the first iteration.

    # TODO: Check if weights here are integer or not
    # TODO: If not, we need to encode and decode it correctly
    # print("current iteration = ", self.current_iteration)
    weights = getWeights(
        previous_weight = self.prevWeight,
        max_iter = self.max_logreg_iterations,
        lr = self.learning_rate,
        trainX = self.trainX[self.current_iteration - 1],
        trainY = self.trainY[self.current_iteration - 1],
        self_id = self.id)
    weights = np.array(weights)

    self.weight_length = len(weights)

    # Send H(k)^{x_i + r_i} to the server
    self.current_base =  self.ro.query(self.current_iteration, self.weight_length)
    # *** This is not secure: each segment should use a different base
    # For accuracy testing purpose, leave it as it is now
    masked_input = [None] * self.weight_length
    for i in range(self.weight_length):
        # exponents[i] = weight[i] + r_i
        intSeg, decSeg = weightFormat.toSegments(weights[i], self.segDigitLength, self.segNumInt, self.segNumDec)
        for j in range(self.segNumInt):
          intSeg[j] = self.finite_field_p.power(self.current_base[i], self.finite_field_q.add(intSeg[j], self.self_r))
        for j in range(self.segNumDec):
          decSeg[j] = self.finite_field_p.power(self.current_base[i], self.finite_field_q.add(decSeg[j], self.self_r))
          
        masked_input[i] = (intSeg, decSeg)
        # masked_input[i] = self.finite_field_p.power(self.current_base[i], self.finite_field_q.add(int((weights[i] + self.offset) * self.multiplier), self.self_r))
        # masked_input[i] = self.finite_field_p.power(self.current_base[i], self.self_r)
    # print(f"client {self.id} - int weights[0] = {int((weights[0]+self.offset) * self.multiplier)}")

    # hash = int.from_bytes(
    #             nh.sha256(self.current_iteration.to_bytes(32, "big")),
    #             "big")
    # base = self.finite_field_p.convert(hash, 2**256)
    # while self.finite_field_p.order(base) * 2 + 1 != self.finite_field_p.getPrime():
    #     base = base + 1
    # self.current_base = base

    return masked_input



  def aggShareSum(self, online_set):
    r_shares_sum = 0
    for peer_id in online_set:
      r_shares_sum = self.finite_field_q.add(r_shares_sum, self.peer_r_share[peer_id])
    r_shares_sum_expo = [0] * self.weight_length
    for i in range(self.weight_length):
      # for peer_id in online_set:
      #   r_shares_sum[i] = self.finite_field_q.add(r_shares_sum[i], self.peer_r_share[peer_id])
      r_shares_sum_expo[i] = self.finite_field_p.power(self.current_base[i], r_shares_sum)

    return r_shares_sum_expo



  def recordTime(self, startTime, categoryName):
      dt_protocol_end = pd.Timestamp('now')
      self.elapsed_time[categoryName] += dt_protocol_end - startTime
      self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))
