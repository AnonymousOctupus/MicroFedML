import nacl.hash as nh

class RandomOracle(object):


  def __init__(self, finite_field):
    super(RandomOracle, self).__init__()
    self.finite_field = finite_field


  # Input the iteration number and the number of bases needed
  def query(self, k, num):
    result = [None] * num

    k = k * num
    for i in range(num):
      hash = int.from_bytes(
                nh.sha256((k + i).to_bytes(32, "big")),
                "big")
      base = self.finite_field.convert(hash, 2**256)
      while self.finite_field.order(base) * 2 + 1 != self.finite_field.getPrime():
      # while self.finite_field.order(base) != self.finite_field.getPrime() - 1:
        base = base + 1
      # result.append(base)
      result[i] = base

    return result



  def leftMostOne(n):
    index = -1
    while n > 0:
      index += 1
      n = n >> 1
    return index
