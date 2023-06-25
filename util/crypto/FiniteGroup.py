

class FiniteGroup(object):
  def __init__(self, n):
    self.n = n

  def getSize(self):
    return self.n

  def add(self, a, b):
    return (a + b) % self.n

  def minus(self, a, b):
    return (a - b) % self.n

  def multAdd(self, list):
    sum = 0
    for v in list:
      sum = sum + v
    sum = sum % n
