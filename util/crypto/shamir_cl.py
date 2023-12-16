import math
import random
import secrets


class Shamir(object):
  """docstring for Shamir."""

  def __init__(self, scale, n = 1):
    super(Shamir, self).__init__()
    self.scale = scale
    self.scalesq = scale * scale
    self.scaletr = self.scalesq * scale
    # self.n = n
    # self.lagrange = [[0] * n] * n
    # for i in range(n):
    #   for j in range(n):
    #     if i == j:
    #       continue
    #     self.lagrange[i][j] = scale * scale * i // (i - j)
    self.precomp = {}
    for i in range(1, n+1):
      self.precomp[i] = 1
      for j in range(1, n+1):
        if j == i:
          continue
        self.precomp[i] = self.precomp[i] * (j - i)

  # s: secret
  # t: threshold
  # holders: the list of indices of holders
  # bound: \ell, the upper bound of the secret
  # scale: scaling the secret to hide the modules
  def secretShare(self, s, t, holders, bound = 1000000000000000000):
    # Uniformly randomly choose coefficients from the field
    a = []
    for i in range(1, t):
      a.append(secrets.randbelow(bound))
    # print(a)

    # calculate the dictionary of shares
    shares = {}
    for x in holders:
      shares[x] = s * self.scale
      temp = x
      for i in range(t - 1):
        # shares[x] = ff.add(shares[x], ff.multiply(a[i], x**(i + 1)))
        shares[x] = shares[x] + a[i] * temp
        temp = temp * x

    # return the shares
    return shares


  # share: dic of (x, s_x)
  # t: threshold
  # not directly used I think
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


  # share: dic of (x, s_x)
  # t: threshold
  def reconstructInExponent(self, shares, t, cl):
    if len(shares) < t:
      return -1

    # ffsub = FiniteField((ff.getPrime() - 1) // 2)
    secret = cl.power_g(0)
    for xj, yj in shares.items():
      temp = yj
      # lagrange = self.scale * self.scale
      # for xm in shares.keys():
      #   if xm == xj:
      #     continue
      #   lagrange = lagrange * xm 
      lagrange = self.scaletr // xj
      # for xm in shares.keys():
      #   if xm == xj:
      #     continue
      #   lagrange = lagrange // (xm - xj)
      lagrange = lagrange // self.precomp[xj]
      temp = cl.power(temp, int(lagrange))
      # temp = cl.power(temp, int(self.lagrange[xj]))
      secret = cl.mul(secret, temp)

    return secret

