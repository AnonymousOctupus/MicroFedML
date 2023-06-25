import math
import random
import secrets
from util.crypto.FiniteField import FiniteField as FiniteField

import gmpy2
from gmpy2 import mpz

class Shamir(object):
  """docstring for Shamir."""

  def __init__(self, ff):
    super(Shamir, self).__init__()
    self.ff = ff
    self.sub_ff = FiniteField(gmpy2.f_div(ff.getPrime(), 2))

  def initCoeff(self, holders):
    ff = self.ff

    self.invModPrime = {}
    for xm in holders:
      if xm == holders[0]:
        continue
      diff1 = ff.subtract(holders[0], xm)
      diff2 = ff.subtract(xm, holders[0])
      self.invModPrime[diff1] = gmpy2.mod(ff.extended_euclid(diff1, ff.getPrime())[1][0], ff.getPrime())
      self.invModPrime[diff2] = gmpy2.mod(ff.extended_euclid(diff2, ff.getPrime())[1][0], ff.getPrime())

    self.coeff_list = {}
    for xm in holders:
      self.coeff_list[xm] = {}
      for xj in holders:
        if xm == xj:
          continue
        # self.coeff_list[xm][xj] = ff.divide(xm, ff.subtract(xm, xj))
        self.coeff_list[xm][xj] = ff.multiply(xm, self.invModPrime[ff.subtract(xm, xj)])

  def initSubCoeff(self, holders):
    # ff = self.ff
    # sub_ff = FiniteField(gmpy2.f_div(ff.getPrime(), 2))
    sub_ff = self.sub_ff

    self.invModPrime_sub = {}
    for xm in holders:
      if xm == holders[0]:
        continue
      diff1 = sub_ff.subtract(holders[0], xm)
      diff2 = sub_ff.subtract(xm, holders[0])
      self.invModPrime_sub[diff1] = gmpy2.mod(sub_ff.extended_euclid(diff1, sub_ff.getPrime())[1][0], sub_ff.getPrime())
      self.invModPrime_sub[diff2] = gmpy2.mod(sub_ff.extended_euclid(diff2, sub_ff.getPrime())[1][0], sub_ff.getPrime())


    self.sub_coeff_list = {}
    for xm in holders:
      self.sub_coeff_list[xm] = {}
      for xj in holders:
        if xm == xj:
          continue
        # self.sub_coeff_list[xm][xj] = sub_ff.divide(xm, sub_ff.subtract(xm, xj))
        self.sub_coeff_list[xm][xj] = sub_ff.multiply(xm, self.invModPrime_sub[sub_ff.subtract(xm, xj)])



  def secretShare(self, s, t, holders):
    ff = self.ff
    # Uniformly randomly choose coefficients from the field
    a = []
    for i in range(1, t):
      a.append(secrets.randbelow(ff.getPrime()))
    # print(a)

    # calculate the dictionary of shares
    shares = {}
    for x in holders:
      shares[x] = s
      temp = x
      for i in range(t - 1):
        # shares[x] = ff.add(shares[x], ff.multiply(a[i], x**(i + 1)))
        shares[x] = self.ff.add(shares[x], self.ff.multiply(a[i], temp))
        temp = self.ff.multiply(temp, x)

    # return the shares
    return shares


  def reconstruct(self, shares, t):
    ff = self.ff

    if len(shares) < t:
      return -1

    secret = 0
    # counter = 0
    for xj, yj in shares.items():
      # if counter == t:
        # break
      lagrange = 1
      for xm in shares.keys():
        if xm == xj:
          continue
        lagrange = ff.multiply(lagrange, self.coeff_list[xm][xj])
      # print(lagrange)
      secret = ff.add(secret, ff.multiply(yj, lagrange))
    return secret


  def reconstructInExponent(self, shares, t):
    ff = self.ff

    if len(shares) < t:
      return -1

    # ffsub = FiniteField(gmpy2.f_div(ff.getPrime(), 2))
    ffsub = self.sub_ff

    secret = 1
    # counter = 0
    for xj, yj in shares.items():
      # if counter == t:
      #   break
      lagrange = 1
      for xm in shares.keys():
        if xm == xj:
          continue
        lagrange = ffsub.multiply(lagrange, self.sub_coeff_list[xm][xj])
        # lagrange = lagrange * (xm / (xm - xj))
      # print("lagrange: ", lagrange)
      secret = ff.multiply(secret, ff.power(yj, lagrange))
      # print(ff.log(secret, 3))
      # counter += 1

    return secret
