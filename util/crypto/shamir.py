import math
import random
import secrets
from util.crypto.FiniteField import FiniteField

def secretShare(s, t, holders, ff):
  # Uniformly randomly choose coefficients from the field
  a = []
  for i in range(1, t):
    a.append(secrets.randbelow(ff.getPrime()))
  # print(a)

  # calculate the dictionary of shares
  shares = {}
  for x in holders:
    shares[x] = s
    temp = x % ff.getPrime()
    for i in range(t - 1):
      # shares[x] = ff.add(shares[x], ff.multiply(a[i], x**(i + 1)))
      shares[x] = ff.add(shares[x], ff.multiply(a[i], temp))
      temp = ff.multiply(temp, x)

  # return the shares
  return shares


def reconstruct(shares, t, ff):
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
      lagrange = ff.multiply(lagrange, ff.divide(xm, ff.subtract(xm, xj)))
    # print(lagrange)
    secret = ff.add(secret, ff.multiply(yj, lagrange))
  return secret


def reconstructInExponent(shares, t, ff):
  if len(shares) < t:
    return -1

  ffsub = FiniteField((ff.getPrime() - 1) // 2)

  secret = 1
  # counter = 0
  for xj, yj in shares.items():
    # if counter == t:
    #   break
    lagrange = 1
    for xm in shares.keys():
      if xm == xj:
        continue
      lagrange = ffsub.multiply(lagrange, ffsub.divide(xm, ffsub.subtract(xm, xj)))
      # lagrange = lagrange * (xm / (xm - xj))
    # print("lagrange: ", lagrange)
    secret = ff.multiply(secret, ff.power(yj, lagrange))
    # print(ff.log(secret, 3))
    # counter += 1

  return secret
