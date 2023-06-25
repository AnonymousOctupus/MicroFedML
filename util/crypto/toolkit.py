from collections import UserList
from util.crypto.FiniteField import FiniteField
from util.crypto.shamir_gmp import Shamir
from util.crypto.RandomOracle import RandomOracle

def ff_ss_ro(prime, user_list, max_input):
    ff_p = FiniteField(prime)
    ff_p.initialize_kangaroo(max_input * len(user_list))
    ff_q = FiniteField(ff_p.getPrime() // 2)
    ss = Shamir(ff_p)
    ss.initCoeff(user_list)
    ss.initSubCoeff(user_list)
    ss_sub = Shamir(ff_q)
    ss_sub.initCoeff(user_list)
    ro = RandomOracle(ff_p)
    
    return [ff_p, ff_q, ss, ss_sub, ro]