from agent.Agent import Agent
from agent.secagg_cl.SA_ServiceAgent import SA_ServiceAgent as ServiceAgent
from message.Message import Message
from util.util import log_print

# from util.crypto.logReg import getWeights, reportStats
import util.crypto.diffieHellman as dh
from util.crypto.FiniteField import FiniteField
from util.crypto.FiniteGroup import FiniteGroup
from util.crypto.RandomOracle import RandomOracle
from util.crypto.shamir_cl import Shamir
import util.formatting.segmenting as segmenting

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
               peer_list = None,
               group_list = None,
               group_id = 0,
               round_time = pd.Timedelta("10s"),
               iterations = 4,
               max_input = 10000,
               seg_length = 3,
               key_length = 256,
               prime = -1,
               class_group = None,
               shamir = None,
               scale = 1,
               pki_pub = None, pki_priv = None,
               peer_pki_pubkeys = None,
               num_clients = None,
               num_groups = None,
               threshold = 10 * 2 // 3 + 1,
               num_subgraphs = None,
               random_state=None):

    # Base class init.
    super().__init__(id, name, type, random_state)


    # Store the client's peer list (subgraph, neighborhood) with which it should communicate.
    self.peer_list = peer_list
    self.group_member_list = group_list
    self.group_id = group_id
    self.group_mapping = {}
    cnt = 1
    for i in self.group_member_list[self.group_id]:
        self.group_mapping[i] = cnt
        cnt += 1

    # Initialize a tracking attribute for the initial peer exchange and record the subgraph size.
    self.num_peers = len(self.peer_list)
    self.threshold = threshold
    self.peer_pki_pubkeys = peer_pki_pubkeys
    self.pki_pub = pki_pub
    self.pki_priv = pki_priv

    # Record the total number of clients participating in the protocol and the number of subgraphs.
    # Neither of these are part of the protocol, or necessary for real-world implementation, but do
    # allow for convenient logging of progress and results in simulation.
    self.num_clients = num_clients
    self.num_groups = num_groups
    self.num_subgraphs = num_subgraphs

    self.neighbor_group_l = group_id - 1
    self.neighbor_group_r = group_id + 1
    if self.neighbor_group_l == -1:
        self.neighbor_group_l = self.num_groups - 1
    if self.neighbor_group_r == self.num_groups:
        self.neighbor_group_r = 0

    # Record the number of iterations the clients will perform.
    self.no_of_iterations = iterations

    # Initialize a dictionary to remember which peers we have heard from during peer exchange.
    self.peer_received = {}

    # Initialize a dictionary to accumulate this client's timing information by task.
    self.elapsed_time = { 'OFFLINE' : pd.Timedelta(0),
                      'ENCRYPTION_R1' : pd.Timedelta(0),
                      'ENCRYPTION_R2' : pd.Timedelta(0) }
    self.message_size = { 'PUBKEY' : 0,
                          'MK_SHARES' : 0,
                          'RH_SHARES' : 0,
                          'MK_SHARES' : 0,
                          'OFFLINE_SIG' : 0,
                          'OFFLINE_MK_SHARES' : 0,
                          'MASKED_INPUT' : 0,
                          'SHARE_SUM' : 0 }
    # self.elapsed_time = { 'DH_OFFLINE' : pd.Timedelta(0), 'DH_ONLINE' : pd.Timedelta(0),
    #                       'TRAINING' : pd.Timedelta(0), 'ENCRYPTION' : pd.Timedelta(0) }


    # Pre-generate this client's local input for each iteration
    # (for the sake of simulation speed).
    self.input = []
    # This is a faster PRNG than the default,
    # for times when we must select a large quantity of randomness.
    self.prng_input = np.random.Generator(np.random.SFC64())
    # For each iteration, pre-generate a random integer as its secret input;
    for i in range(iterations):
        self.input.append(self.prng_input.integers(low = 0, high = max_input));
    self.max_sum = max_input * len(self.peer_list)
    self.seg_length = seg_length
    self.finite_group = FiniteGroup(self.max_sum)
    self.seg_num = segmenting.segNum(self.max_sum, seg_length)

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
    self.mk_pub = None
    self.mk_priv = None
    # Dictionaries of the symmetric encryption keys
    self.peer_ek = {}
    self.peer_mk = {}
    self.peer_secret_box = {}
    # Record the self mask
    self.self_r = 0
    # Dictionary of the other users' shares
    self.peer_mk_share_l = {}
    self.peer_mk_share_r = {}
    self.peer_r_share = {}
    self.peer_h_share = {}

    self.cl = class_group
    self.ss = shamir
    self.scale = scale


    ### ADD DIFFERENTIAL PRIVACY CONSTANTS AND CONFIGURATION HERE, IF NEEDED.
    #
    #


    # Iteration counter.
    self.current_iteration = 0
    self.current_base = []
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
        # self.kernel.custom_state['dh_offline'] = pd.Timedelta(0)
        # self.kernel.custom_state['dh_online'] = pd.Timedelta(0)
        # self.kernel.custom_state['training'] = pd.Timedelta(0)
        # self.kernel.custom_state['encryption'] = pd.Timedelta(0)

        self.kernel.custom_state['bdw_pubkey'] = 0
        self.kernel.custom_state['bdw_mk_shares'] = 0
        self.kernel.custom_state['bdw_rh_shares'] = 0
        self.kernel.custom_state['bdw_offline_sig'] = 0
        self.kernel.custom_state['bdw_offline_mk_shares'] = 0
        self.kernel.custom_state['bdw_masked_input'] = 0
        self.kernel.custom_state['bdw_share_sum'] = 0

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
    self.kernel.custom_state['bdw_mk_shares'] += self.message_size['MK_SHARES']
    self.kernel.custom_state['bdw_rh_shares'] += self.message_size['RH_SHARES']
    self.kernel.custom_state['bdw_offline_sig'] += self.message_size['OFFLINE_SIG']
    self.kernel.custom_state['bdw_offline_mk_shares'] += self.message_size['OFFLINE_MK_SHARES']
    self.kernel.custom_state['bdw_masked_input'] += self.message_size['MASKED_INPUT']
    self.kernel.custom_state['bdw_share_sum'] += self.message_size['SHARE_SUM']

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
        dt_start = pd.Timestamp('now')
        # Generate DH keys.
        self.ek_pub, self.ek_priv = nb.crypto_kx_keypair()
        self.mk_pub_l, self.mk_priv_l = nb.crypto_kx_keypair()
        self.mk_pub_r, self.mk_priv_r = nb.crypto_kx_keypair()

        # Sign the public key with the signing key
        # signature = self.pki_priv.sign(
        #                             self.ek_pub + self.mk_pub_l + self.mk_pub_r,
        #                             encoder=Base64Encoder)

        self.recordTime(dt_start, 'OFFLINE')
        # Send one message to the server,
        # including id, ek_pub, mk_pub_l/r, sig
        # Let the server check the signature and forward to other peers
        self.sendMessage(self.serviceAgentID,
                        Message({ "msg" : "SETUP_PUBKEYS",
                          "iteration": self.current_iteration,
                          "sender" : self.id,
                          "sender_group": self.group_id,
                          "ek_pub" : self.ek_pub,
                          "mk_pub_l" : self.mk_pub_l,
                          "mk_pub_r" : self.mk_pub_r,
                        #   "sig"    : signature,
                          }),
                        tag = "comm_offline")
        self.recordBandwidth(self.ek_pub, 'PUBKEY')
        self.recordBandwidth(self.mk_pub_l, 'PUBKEY')
        self.recordBandwidth(self.mk_pub_r, 'PUBKEY')

    else:
        self.aggMaskedInput()
        # masked_input = self.aggMaskedInput()
        #
        # self.recordTime(dt_start, 'ENCRYPTION_R1')
        # self.sendMessage(self.serviceAgentID,
        #         Message({ "msg" : "AGG_MASKED_INPUT",
        #                 "iteration": self.current_iteration,
        #                 "sender": self.id,
        #                 "sender_group": self.group_id,
        #                 "masked_input": masked_input,
        #                 }),
        #         tag = "comm_online")




  def receiveMessage (self, currentTime, msg):
    super().receiveMessage(currentTime, msg)
    # print("client", self.id, "receives message", msg.body['msg'])
    # print(f"\t self iteration: {self.current_iteration}")

    # Setup phase: Round 2
    # Receiving ek_pub with signatures of other clients from the server
    if msg.body['msg'] == "SETUP_PUBKEYS" and self.current_iteration == 0:

        # Record start of message processing.
        dt_start = pd.Timestamp('now')

        # Record all symmetric keys for each peer client
        for peer_id, peer_pubkeys in msg.body['pubkeys'].items():

            # Verify the signature of the sender client
            # If invalid, then the server is corrupt, abort.
            # signature = peer_pubkeys[3]
            # pki_pub = VerifyKey(self.peer_pki_pubkeys[peer_id], encoder=Base64Encoder)
            # # signature = signature[:-1] + bytes([int(signature[-1]) ^ 1])
            # pki_pub.verify(signature, encoder=Base64Encoder)

            # Key agreement to get the symmetric encryption keys
            # Store the agreed keys
            ek = dh.keyexchange(
                    peer_id, self.id, self.ek_pub, self.ek_priv, peer_pubkeys[0])
            self.peer_ek[peer_id] = ek
            # print("symmetric key between", self.id, "and", peer_id, ":", (self.peer_ek[peer_id]))
            byte_key = ek.to_bytes(self.key_length, 'big')
            self.peer_secret_box[peer_id] = nacl.secret.SecretBox(byte_key)

            # Key agreement to get the symmetric masking keys
            if peer_id in self.group_member_list[self.neighbor_group_l]:
                mk = dh.keyexchange(
                        peer_id, self.id, self.mk_pub_l, self.mk_priv_l, peer_pubkeys[2])
                self.peer_mk[peer_id] = mk
            if peer_id in self.group_member_list[self.neighbor_group_r]:
                mk = dh.keyexchange(
                        peer_id, self.id, self.mk_pub_r, self.mk_priv_r, peer_pubkeys[1])
                self.peer_mk[peer_id] = mk

        # Check if the set of users I have done keyexchange with is large enough
        # If not, abort;
        # TODO:

        # Generate the shares of the secret mutual mask key (among group members)
        # l_shares = shamir.secretShare(s = int.from_bytes(self.mk_priv_l, "big"),
        #                             t = self.threshold,
        #                             holders = self.group_member_list[self.neighbor_group_l],
        #                             ff = self.finite_field_q)

        # r_shares = shamir.secretShare(s = int.from_bytes(self.mk_priv_r, "big"),
        #                             t = self.threshold,
        #                             holders = self.group_member_list[self.neighbor_group_r],
        #                             ff = self.finite_field_q)

        holders_l = range(1, len(self.group_member_list[self.neighbor_group_l]) + 1)
        holders_r = range(1, len(self.group_member_list[self.neighbor_group_r]) + 1)
        l_shares = self.ss.secretShare(s = int.from_bytes(self.mk_priv_l, "big"),
                                    t = self.threshold,
                                    holders = holders_l)
                                    # holders = self.group_member_list[self.neighbor_group_l])

        r_shares = self.ss.secretShare(s = int.from_bytes(self.mk_priv_r, "big"),
                                    t = self.threshold,
                                    holders = holders_r)
                                    # holders = self.group_member_list[self.neighbor_group_r])


        # Encrypt the shares of the secret key
        share_cipher_l = {}
        # for peer_id in self.group_member_list[self.neighbor_group_l]:
        for i in holders_l:
            peer_id = self.group_member_list[self.neighbor_group_l][i - 1]
            if peer_id not in self.peer_secret_box:
                continue
            byte_share = int(l_shares[i]).to_bytes(256, "big")
            # share_cipher_l[i] = self.peer_secret_box[peer_id].encrypt(byte_share)
            # byte_share = int(l_shares[peer_id]).to_bytes(256, "big")
            share_cipher_l[peer_id] = self.peer_secret_box[peer_id].encrypt(byte_share)

        share_cipher_r = {}
        # for peer_id in self.group_member_list[self.neighbor_group_r]:
        for i in holders_r:
            peer_id = self.group_member_list[self.neighbor_group_r][i - 1]
            if peer_id not in self.peer_secret_box:
                continue
            byte_share = int(r_shares[i]).to_bytes(256, "big")
            # share_cipher_r[i] = self.peer_secret_box[peer_id].encrypt(byte_share)
            # byte_share = int(r_shares[peer_id]).to_bytes(256, "big")
            share_cipher_r[peer_id] = self.peer_secret_box[peer_id].encrypt(byte_share)


        self.recordTime(dt_start, 'OFFLINE')
        # Send the mutual mask keys and ciphers of shares to the server
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SETUP_MK_SHARES_CIPHER",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "sender_group": self.group_id,
                        "mk_shares_cipher_l": share_cipher_l,
                        "mk_shares_cipher_r": share_cipher_r,
                        }),
                tag = "comm_offline")
        self.recordBandwidth(share_cipher_l, 'MK_SHARES')
        self.recordBandwidth(share_cipher_r, 'MK_SHARES')



    # Setup phase: Round 3
    # Receiving the mutual mask key exchange information from all peers
    elif msg.body['msg'] == "SETUP_MK_SHARES_CIPHER":
        # print("Client", self.id, "receives shares for mk_priv")

        dt_start = pd.Timestamp('now')

        mk_shares_cipher_l = msg.body['mk_shares_cipher_l']
        mk_shares_cipher_r = msg.body['mk_shares_cipher_r']

        # Check that for any group,
        # It receives at least t messages
        # TODO:

        # Decrypt the ciphers of shares and store
        # Also calculates the h-mask
        h = 0
        for peer_id in mk_shares_cipher_l:
            byte_share = self.peer_secret_box[peer_id].decrypt(mk_shares_cipher_l[peer_id])
            self.peer_mk_share_l[peer_id] = int.from_bytes(byte_share, "big")
            # h = self.finite_field_q.add(h, self.peer_mk[peer_id])
            h = h + self.peer_mk[peer_id]
        for peer_id in mk_shares_cipher_r:
            byte_share = self.peer_secret_box[peer_id].decrypt(mk_shares_cipher_r[peer_id])
            self.peer_mk_share_r[peer_id] = int.from_bytes(byte_share, "big")
            # h = self.finite_field_q.subtract(h, self.peer_mk[peer_id])
            h = h - self.peer_mk[peer_id]
        self.self_h = h

        # Calculates the segments of h
        # self.self_h_segs = segmenting.segment(h,
        #                                     self.seg_num,
        #                                     self.seg_length)

        # Secret share the segments of h-mask among the group members
        # h_shares = shamir.secretShare(s = self.self_h,
        #                         t = self.threshold,
        #                         holders = self.group_member_list[self.group_id],
        #                         ff = self.finite_field_q)

        holders = list(range(1, len(self.group_member_list[self.group_id]) + 1))
        h_shares = self.ss.secretShare(s = self.self_h,
                                t = self.threshold,
                                holders = holders)
                                # holders = self.group_member_list[self.group_id])
        # h_seg_shares = []
        # for rho in  range(self.seg_num):
        #     h_seg_shares.append(shamir.secretShare(s = self.self_h_segs[rho],
        #                             t = self.threshold,
        #                             holders = self.group_member_list[self.group_id],
        #                             ff = self.finite_field_q))

        # Generate the self mask (r-mask)
        self.self_r = secrets.randbelow(1 << 256)
        # self.self_r = self.id

        # Secret shares the r-mask among the group members
        # r_shares = shamir.secretShare(s = self.self_r,
        #                             t = self.threshold,
        #                             holders = self.group_member_list[self.group_id],
        #                             ff = self.finite_field_q)
        r_shares = self.ss.secretShare(s = self.self_r,
                                    t = self.threshold,
                                    holders = holders)
                                    # holders = self.group_member_list[self.group_id])

        # Encrypt each r-share and h-share with the symmetric key
        r_share_ciphers = {}
        h_share_ciphers = {}
        # for peer_id in self.group_member_list[self.group_id]:
        for i in holders:
            peer_id = self.group_member_list[self.group_id][i - 1]
            # byte_share = int(r_shares[peer_id]).to_bytes(256, "big")
            byte_share = int(r_shares[i]).to_bytes(256, "big", signed = True)
            r_share_ciphers[peer_id] = self.peer_secret_box[peer_id].encrypt(byte_share)
            # r_share_ciphers[i] = self.peer_secret_box[peer_id].encrypt(byte_share)

            # byte_share = int(h_shares[peer_id]).to_bytes(256, "big")
            byte_share = int(h_shares[i]).to_bytes(256, "big", signed = True)
            h_share_ciphers[peer_id] = self.peer_secret_box[peer_id].encrypt(byte_share)
            # h_share_ciphers[i] = self.peer_secret_box[peer_id].encrypt(byte_share)

        self.recordTime(dt_start, 'OFFLINE')
        # Send the encrypted shares to the server
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SETUP_RH_SHARES_CIPHER",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "sender_group": self.group_id,
                        "r_share_ciphers": r_share_ciphers,
                        "h_share_ciphers": h_share_ciphers,
                        }),
                tag = "comm_offline")
        self.recordBandwidth(r_share_ciphers, 'RH_SHARES')
        self.recordBandwidth(h_share_ciphers, 'RH_SHARES')


    # Setup phase: Round 4
    # Receiving encrypted r and h shares from the neighbors
    # Sending the offline set with the signature to members of neighboring groups
    elif msg.body['msg'] == "SETUP_RH_SHARES_CIPHER" and self.current_iteration == 0:
        dt_start = pd.Timestamp('now')

        r_ciphers = msg.body['r_shares_cipher']
        h_ciphers = msg.body['h_shares_cipher']

        # If receiving less than t shares, abort
        if len(r_ciphers) < self.threshold:
            print("client", self.id, "receives less than t shares")
            self.current_iteration = -1
            return

        # Decrypt the shares with corresponding symm key
        for peer_id in r_ciphers:
            byte_share = self.peer_secret_box[peer_id].decrypt(r_ciphers[peer_id])
            self.peer_r_share[peer_id] = int.from_bytes(byte_share, "big", signed = True)
            byte_share = self.peer_secret_box[peer_id].decrypt(h_ciphers[peer_id])
            self.peer_h_share[peer_id] = int.from_bytes(byte_share, "big", signed = True)

        # Generate offline list: users sent ek_pub but did not share r and h
        offline = []
        for neighbor_id in self.group_member_list[self.group_id]:
            if neighbor_id in self.peer_ek and neighbor_id not in self.peer_h_share:
                offline.append(neighbor_id)
        # Sign and send the offline set of group d to the server
        # signature = self.pki_priv.sign(offline, encoder=Base64Encoder)

        self.recordTime(dt_start, 'OFFLINE')
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SETUP_OFFLINE_SET",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "sender_group": self.group_id,
                        "offline_set": offline,
                        # "sig": signature,
                        }),
                tag = "comm_offline")
        self.recordBandwidth(offline, 'OFFLINE_SIG')


    # Setup phase: Round 5
    # Receiving the offline set from the other clients
    # Sending the shares of mk_priv of offline clients to the server
    elif msg.body['msg'] == "SETUP_OFFLINE_SET" and self.current_iteration == 0:
        dt_start = pd.Timestamp('now')

        offline_count_l = {}
        for peer_id, offline_set in msg.body['offline_sets_l'].items():
            # TODO: Verify the signature

            for offline_id in offline_set:
                if offline_id not in offline_count:
                    offline_count_l[offline_id] = 1
                else:
                    offline_count_l[offline_id] += 1
        # Collect shares of mk_priv of offline users
        mk_priv_shares_l = {}
        for peer_id, c in offline_count_l:
            if c >= self.threshold:
                mk_priv_shares_l[peer_id] = self.peer_mk_share[peer_id]

        offline_count_r = {}
        for peer_id, offline_set in msg.body['offline_sets_r'].items():
            # TODO: Verify the signature

            for offline_id in offline_set:
                if offline_id not in offline_count:
                    offline_count_r[offline_id] = 1
                else:
                    offline_count_r[offline_id] += 1
        # Collect shares of mk_priv of offline users
        mk_priv_shares_r = {}
        for peer_id, c in offline_count_r:
            if c >= self.threshold:
                mk_priv_shares_r[peer_id] = self.peer_mk_share[peer_id]

        self.recordTime(dt_start, 'OFFLINE')
        # Send the shares of mk_priv of offline nodes to the server
        self.sendMessage(self.serviceAgentID,
                Message({ "msg" : "SETUP_MK_PRIV_SHARES",
                        "iteration": self.current_iteration,
                        "sender": self.id,
                        "sender_group": self.group_id,
                        "mk_priv_shares_l": mk_priv_shares_l,
                        "mk_priv_shares_r": mk_priv_shares_r,
                        }),
                tag = "comm_offline")
        self.recordBandwidth(mk_priv_shares_l, 'OFFLINE_MK_SHARES')
        self.recordBandwidth(mk_priv_shares_r, 'OFFLINE_MK_SHARES')

        # Enter iteration 1.
        self.current_iteration += 1


    # Signal from the server that the Aggregation phase starts
    elif msg.body['msg'] == "AGG_START" and self.current_iteration != 0:
        dt_start = pd.Timestamp('now')

        self.aggMaskedInput()


    # Aggregation phase: Round 2
    # Receiving the online set from the server
    elif msg.body['msg'] == "AGG_ONLINE_SET" and self.current_iteration != 0:
        online_set = msg.body["online_set"]
        self.aggShareSum(online_set)



    # Aggregation phase: Round 3
    # Receiving the output from the server
    elif msg.body['msg'] == "AGG_OUTPUT" and self.current_iteration != 0:
        # End of the iteration
        # In practice, the client can do calculations here
        # based on the output of the server
        # In this simulation we just log the information
        output = msg.body['output']

        # print(f"User {self.id} completes iteration {self.current_iteration}")

        # Reset temp variables for each iteration

        if self.id == 1: print (f"Protocol iteration {self.current_iteration} complete.")

        # Enter next iteration
        self.current_base = []
        self.current_iteration += 1
        # Start a new iteration
        if self.current_iteration > self.no_of_iterations:
          # End the protocol
          print("client", self.id, "input list:", self.input)
          return
        self.aggMaskedInput()


        log_print ("Client weights received for iteration {} by {}: {}", self.current_iteration, self.id, output)




  #================= Round logics =================#
  def aggMaskedInput(self):
    dt_start = pd.Timestamp('now')
    masked_input = self.cl.power_g(self.self_r)
    # print(f"h mask: {self.self_h}")
    masked_input = self.cl.mul(masked_input, self.cl.power_g(self.self_h))
    masked_input = self.cl.power(masked_input, self.scale ** 3)
    masked_input = self.cl.mul(masked_input, self.cl.power_f(self.input[self.current_iteration - 1]))

    # masked_input = self.finite_field_q.add(self.input[self.current_iteration - 1], self.self_h)
    # masked_input = self.finite_field_q.add(masked_input, self.self_r)

    # masked_input = self.finite_field_p.power(self.current_base, masked_input)

    self.recordTime(dt_start, 'ENCRYPTION_R1')
    self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "AGG_MASKED_INPUT",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "sender_group": self.group_id,
                    "masked_input": masked_input,
                    }),
            tag = "comm_online_r1")
    self.recordBandwidth(masked_input, 'MASKED_INPUT')
    # print(f"User {self.id} sends masked input in iteration {self.current_iteration}")



  def aggShareSum(self, online_set):
    dt_start = pd.Timestamp('now')

    shares_sum = 0
    for peer_id in self.peer_h_share:
      if peer_id in online_set:
        #   shares_sum = self.finite_field_q.add(shares_sum, self.peer_r_share[peer_id])
          shares_sum += self.peer_r_share[peer_id]
      else:
        #   shares_sum = self.finite_field_q.subtract(shares_sum, self.peer_h_share[peer_id])
          shares_sum -= self.peer_h_share[peer_id]
    # shares_sum = self.finite_field_p.power(self.current_base, shares_sum)
    shares_sum = self.cl.power_g(shares_sum)

    self.recordTime(dt_start, 'ENCRYPTION_R2')
    self.sendMessage(self.serviceAgentID,
            Message({ "msg" : "AGG_SHARES_SUM",
                    "iteration": self.current_iteration,
                    "sender": self.id,
                    "sender_group": self.group_id,
                    "shares_sum": shares_sum,
                    }),
            tag = "comm_online_r2")
    self.recordBandwidth(shares_sum, 'SHARE_SUM')


# ======================== UTIL ========================
  def recordTime(self, startTime, categoryName):
     # Accumulate into offline setup.
     dt_protocol_end = pd.Timestamp('now')
     self.elapsed_time[categoryName] += dt_protocol_end - startTime
     self.setComputationDelay(int((dt_protocol_end - startTime).to_timedelta64()))

  def recordBandwidth(self, msgobj, categoryName):
      self.message_size[categoryName] += asizeof.asizeof(msgobj)