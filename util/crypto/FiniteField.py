import math
import random

import gmpy2
from gmpy2 import mpz

class FiniteField(object):
  p = mpz(0)
  log_upper_bound = 0
  log_upper_bound_length = 0
  kangaroo_map = {}
  kangaroo_steps = None

  def __init__(self, p = -1, log_upper_bound = 10000):
    if p == -1:
      p_path = "./util/crypto/parameters/prime"
      f = open(p_path)
      p_string = f.read().split()
      f.close()

      # self.p = 0
      for x in p_string:
        self.p = mpz(self.p * (2 ** 32) + int(x, 16))
    else:
      self.p = mpz(p)

    self.q = mpz(gmpy2.f_div(self.p, 2))
    self.log_upper_bound = log_upper_bound
    self.log_upper_bound_length = log_upper_bound.bit_length()

    # print("prime is ", self.p)
    # For discrete log (giant-baby step)
    # self.m = int(math.ceil(math.sqrt(self.p - 1)))

  def getPrime(self):
    return self.p

  # a + b
  def add (self, a, b):
    return gmpy2.f_mod(a + b, self.p)

  # a - b
  def subtract (self, a, b):
    return gmpy2.f_mod(a - b, self.p)

  # a * b
  def multiply (self, a, b):
    return gmpy2.f_mod(a * b, self.p)

  # a / b
  def divide (self, a, b):
    invModPrime = gmpy2.mod(self.extended_euclid(b, self.p)[1][0], self.p)
    return self.multiply(a, invModPrime)

  # a^b
  # TODO: will this be slow?
  def power (self, a, b):
    return gmpy2.powmod(a, b, self.p)

  # log_b a
  # Works for small exponents
  def log (self, a, b):
    # result = -1
    # if self.log_upper_bound_length > 17:
    #   for i in range(3):
    #     print(f"Long log: using kangaroo - time {i}")
    #     result = self.log_kangaroo(b, a)
    #     if result > -1:
    #       break
    # if result > -1:
    #   return result
    # print("Using brute force to calculate discrete log")
    return self.log_brute_force(a, b)

      # Brute force
    # result = 0
    # temp = 1
    # while temp != a and result < self.p:
    #   result += 1
    #   temp = (temp * b) % self.p
    #   if (result > 1 << 33):
    #     # print(a, b)
    #     break;
    # return result

  # order of a
  def order (self, a):
    if self.multiply(a, a) == 1:
      return 2
    if self.power(a, self.q) == 1:
      return self.q
    else:
      return self.p - 1

  def convert (self, a, range):
    return (a * self.p) // range





#============== Different log algorithms ===============#
# Calculate log_b a
  def log_brute_force (self, a, b):

    result = 0
    temp = 1
    while temp != a and result < self.p:
      result += 1
      temp = gmpy2.f_mod((temp * b), self.p)
      if (result > 1 << 33):
        # print(a, b)
        break
    return result

  # Calculate log_b a: allow negative results
  def log_brute_force_with_neg (self, a, b):
    # print("a =", a)

    result = 0
    negres = 0
    temp = 1
    negtemp = 1
    # inv_b = gmpy2.mod(self.extended_euclid(b, self.p)[1][0], self.p)
    inv_b = gmpy2.invert(b, self.p)
    while True:
      # print("currently trying", result)
      if temp == a:
        return result
      if negtemp == a:
        return negres
      result += 1
      temp = gmpy2.f_mod((temp * b), self.p)
      negres -= 1
      negtemp = gmpy2.f_mod((negtemp * inv_b), self.p)
      if (result > 1 << 33):
        # print(a, b)
        break
    return result


  def initialize_kangaroo(self, max_log):
    self.log_upper_bound = max_log
    # self.log_upper_bound_length = length
    self.log_upper_bound_length = max_log.bit_length()
    # print(self.log_upper_bound_length)

    self.kangaroo_steps = []
    for i in range(self.log_upper_bound_length):
      self.kangaroo_steps.append(1 << i)
    # temp = 1
    # p_length = self.p.bit_length()
    # upbd = self.p >> (p_length // 8)
    # upbd = upbd >> 8
    # # print(upbd.bit_length())
    # while temp < (upbd >> 8):
    #   self.kangaroo_steps.append(temp)
    #   temp = temp << 1
      # print(temp.bit_length())



  # kangaroo algorithm
  def log_kangaroo (self, alpha, beta):
    if alpha == beta:
      return 1
    if beta == 1:
      return 0

    # N = int(math.sqrt(self.log_upper_bound))
    N = 128 * (self.log_upper_bound >> (self.log_upper_bound_length //2))
    checkpoint = 8
    # S = upperbound
    ff = FiniteField(self.q)

    tamed = {}
    x = self.power(alpha, self.log_upper_bound)
    curr_log_x = self.log_upper_bound
    wild = {}
    y = beta
    curr_log_y = 0

    tamed_set = set()
    wild_set = set()
    for i in range(N):
      tamed[x] = curr_log_x
      tamed_set.add(x)
      if x not in self.kangaroo_map:
        self.kangaroo_map[x] = random.randint(0, len(self.kangaroo_steps) - 1)
      step = self.kangaroo_steps[self.kangaroo_map[x]]
      x = self.multiply(x, self.power(alpha, step))
      curr_log_x = curr_log_x + step

      wild[y] = curr_log_y
      wild_set.add(y)
      if y not in self.kangaroo_map:
        self.kangaroo_map[y] = random.randint(0, len(self.kangaroo_steps) - 1)
      step = self.kangaroo_steps[self.kangaroo_map[y]]
      y = self.multiply(y, self.power(alpha, step))
      curr_log_y = curr_log_y + step

      if i % checkpoint == 0:
        # common = list(set(tamed.keys()) & set(wild.keys()))
        common = list(tamed_set & wild_set)
        if len(common) > 0:
          # print("succeed, ", tamed[common[0]] - wild[common[0]])
          return tamed[common[0]] - wild[common[0]]



    # while curr_log_y <= self.log_upper_bound + curr_log_x:
    #   if y in tamed:
    #     return tamed[y] - curr_log_y
    #   if y not in self.kangaroo_map:
    #     self.kangaroo_map[y] = random.randint(0, len(self.kangaroo_steps) - 1)
    #   step = self.kangaroo_steps[self.kangaroo_map[y]]
    #   y = self.multiply(y, self.power(alpha, step))
    #   curr_log_y = curr_log_y + step

    return -1


    # x = [0] * (N + 1)
    # d = 0
    # randmap = {}
    #
    # x[0] = self.power(alpha, upperbound)
    # for i in range(N):
    #   if x[i] not in randmap:
    #     randmap[x[i]] = random.randint(0, S)
    #   x[i + 1] = self.multiply(x[i], self.power(alpha, randmap[x[i]]))
    #   d += randmap[x[i]]
    #
    # y = beta
    # d_i = 0
    # while d_i <= upperbound + d:
    #   if y == x[N - 1]:
    #     return (upperbound + d - d_i) % self.p
    #   if y not in randmap:
    #     randmap[y] = random.randint(0, S)
    #   d_i = d_i + randmap[y]
    #   y = self.multiply(y, self.power(alpha, randmap[y]))
    #
    # return -1

  def randMap(self, l, S):
    random = hash(l)
    return random & S


# TODO: Finish the algorithm
  def log_small_interval (self, a, b, l):
    R = 128
    W = 128
    # W = 524288
    T = 896
    N = 7168

    p = self.p
    g = self.p // l
    g = (g * g) % p

    s = [];
    slog = [];
    for i in range(R):
      slog.append(random.randint(0, (1<<46) - 1) // W);
    for i in range(R):
      s.append(self.power(g, slog[i]));

    totalnumsteps = 0
    numsteps = 0

    table = {}
    tabledone = 0
    while tabledone < N:
      wlog = random.randint(0, 1 << 48 - 1)
      w = self.power(g, wlog)

      for log in range(8 * W):
        if self.distinguished(w, W):
          for i in range(tabledone):
            if table[i][0] == w:
              break
          if i < tabledone:
            table[i][2] += 4 * W + numsteps
          else:
            table[tabledone] = [w, wlog, 4*w + numsteps]
            tabledone += 1;
          numsteps = 0
          break

        h = self.hash(w, R)
        wlog = wlog + slog[h]
        w = (w * s[h]) % p
        numsteps += 1
        totalnumsteps += 1

    # TODO: Sort by weight

    totalnumsteps = 0
    hlog = random.randint(0, (1<<48) - 1)
    h = self.power(g, hlog)
    # print(hlog)

    while True:
      numsteps = 0

      wdist = random.randint(0, (1<<40) - 1)
      w = self.multiply(h, self.power(g, wdist))
      for loop in range(8*W):
        if self.distinguished(w, W):
          pointer = T
          for i in range(T):
            if table[i][0] >= w:
              pointer = i
              break
          if pointer < T:
            if table[pointer][0] == w:
              wdist = table[pointer][1] - wdist
          break
        h = self.hash(w, R)
        wdist = wdist + slog[h]
        w = self.multiply(w, s[h])
        numsteps += 1
        totalnumsteps += 1

      if this.power(g, wdist) == h:
        return wdist

  # hash & distinguished point
  def hash(self, w, R):
    return w & (R - 1)
  def distinguished(self, w, W):
    return (w & (W - 1)) == 0

  # tablesort
  def getWeight(self, tuple):
    return tuple[2]
  # tablesort2
  def getValue(self, tuple):
    return tuple[1]















# ==================== Should be removed ====================

# Incorrect
  def log_rho(self, a, b):
    # Generate a small table of size T
    table = {}
    T = self.log_upper_bound # TODO: Change size T
    temp = b
    temp_x = 0
    while len(table) < T:
      x = random.randint()
      temp = self.multiply(temp, self.power(b, x))
      temp_x += x
      table[temp] = temp_x

    # Original Rho method
    P = self.p
    Q = P // 2
    H = a
    G = b

    x = G
    a = 0
    b = 0

    X = x
    A = a
    B = b

    for i in range(1, self.log_upper_bound):
      # Hedgehog
      x, a, b = self.xab(x, a, b, G, H)
      # Rabbit
      X, A, B = self.xab(X, A, B, G, H)
      X, A, B = self.xab(X, A, B, G, H)

      # print(x, a, b)

      if x == X:
        break

    nom = a-A
    denom = B-b

    # print(nom, denom)

    # It is necessary to compute the inverse to properly compute the fraction mod q
    res = (self.extended_euclid(denom, Q)[0] * nom) % Q

    # I know this is not good, but it does the job...
    if self.power(b, res) == a:
      return res

    return res + Q


# Incorrect
  def log_giant_baby (self, a, b):
    x = dict()
    for j in range(self.m):
      x[pow(b, j, self.p)] = j
    invModPrime = self.extended_euclid(b, self.p)[1][0] % self.p

    c = pow(invModPrime, self.m, self.p)
    q = a
    for i in range (1, m):
      q = (q * c) % self.p
      if q in x:
        result = (i * m + x[q]) % self.p
        break

    if pow(b, result, self.p) == b:
      return result


  # Given x, y, return (gcd(x, y), r, s)
  # where r * x + s * y = gcd(x, y)
  def extended_euclid(self, x, y):
    (a, b, aa, bb)=(0, 1, 1, 0)
    while y !=0:
        (q, r) = gmpy2.f_divmod(x, y)
        (a, b, aa, bb) = (aa - q * a, bb - q * b, a, b)
        (x, y) = (y, r)
    return (x, (aa, bb))


  def xab(self, x, a, b, G, H):
    sub = x % 3
    P = self.p
    # Q = P // 2
    Q = self.q

    if sub == 0:
      x = gmpy2.f_mod(x*G, P)
      a = gmpy2.f_mod((a+1), Q)

    if sub == 1:
      x = gmpy2.f_mod(x * H, P)
      b = gmpy2.f_mod((b + 1), Q)

    if sub == 2:
      x = gmpy2.f_mod(x*x, P)
      a = gmpy2.f_mod(a*2, Q)
      b = gmpy2.f_mod(b*2, Q)

    return x, a, b
