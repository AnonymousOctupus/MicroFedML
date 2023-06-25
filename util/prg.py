# import numpy as np
import random

def prg(seed = -1, low = 0, high = (1 << 31) - 1, length = 1):
    res = 0

    if seed != -1:
      seed = seed % (2^32)
      # np.random.seed(seed)
      random.seed(seed)
    res = [0] * length
    for i in range(length):
      # res.append(np.random.randint(low, high))
      # res[i] = np.random.randint(low, high)
      res[i] = random.randint(low, high)

    return res

# print(prg(200), prg(200))
