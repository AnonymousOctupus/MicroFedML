import math
import random
import secrets
import util.crypto.class_group.classgroup as classgroup
import util.formatting.segmenting as seg

class ClassGroup:
  """docstring for ClassGroup."""

  def __init__(self, conv_bits = 30, conv_list_len = 20):
    super(ClassGroup, self).__init__()
    self.cl = classgroup.ClassGroup()
    self.g = self.cl.power_g(1)
    self.f = self.cl.power_f(1)
    self.zero = self.cl.power_g(0)

    self.conv_bits = conv_bits

    self.conv_exp_list_g = [None] * conv_list_len
    # self.conv_exp_list_h = [None] * conv_list_len
    e = 1 << conv_bits
    self.conv_exp_list_g[0] = self.g
    for i in range(conv_list_len - 1):
      self.conv_exp_list_g[i + 1] = self.cl.power(self.conv_exp_list_g[i], e - 1)
      self.conv_exp_list_g[i + 1] = self.cl.mul(self.conv_exp_list_g[i + 1], self.g)

  def mul(self, a1, a2):
    return self.cl.mul(a1, a2)
  
  def div(self, a1, a2):
    return self.cl.div(a1, a2)

  def power(self, a, e):
    isNeg = False
    if e < 0:
      isNeg = True
      e = -e

    segs = seg.segment(e, length = self.conv_bits)
    segs.reverse()
    res = self.cl.power_g(0)
    for i in range(len(segs)):
      res = self.cl.power(res, 1 << self.conv_bits)
      temp = self.cl.power(a, segs[i])
      if isNeg:
        res = self.cl.div(res, temp)
      else:
        res = self.cl.mul(temp, res)
    
    # if isNeg:
    #   res = self.div(self.zero, res)

    return res
  
  def power_g(self, e):
    # segs = seg.segment(e, length = self.conv_bits)
    # segs.reverse()
    # res = self.cl.power_g(0)
    # for i in range(len(segs)):
    #   res = self.cl.power(res, 1 << self.conv_bits)
    #   temp = self.cl.power(self.g, segs[i])
    #   res = self.cl.mul(temp, res)
    # return res
    return self.power(self.g, e)

  def power_f(self, e):
    return self.power(self.f, e)
    # return self.cl.power_f(e)
  
  def log_f_mpz(self, a):
    return self.cl.log_f(a)

  def log_f(self, a):
    es = self.cl.log_f(a)
    e = int(self.cl.to_string_mpz(es))
    return e
  
  def to_string_mpz(self, m):
    return self.cl.to_string_mpz(m)
  
  def to_string_qfi(self, m):
    return self.cl.to_string_qfi(m)
  

#   def toSegments(self, e):
#     # wstr = str(weight) 
#     wstr = numpy.format_float_positional(weight)
#     negSign = wstr[0] == '-'
#     if negSign:
#         wstr = wstr[1:]
#     #   print("is weight negative?", negSign)

#     #   intDec = intDec.rstrip("0")
#     intDec = wstr.split(".") # integer part and decimal part
#     intPart = int(intDec[0])

#     if len(intDec) == 1:
#         intDec.append("0")
#     decLength = len(intDec[1])
#     padding = "0" * (segDigitLength * segNumDec - decLength)
#     intDec[1] = intDec[1] + padding
#     decSeg = []
#     for i in range(segNumDec):
#         # print(intDec[1])
#         # print(i*segDigitLength, (i+1)*segDigitLength)
#         # print(intDec[1][i*segDigitLength, (i+1)*segDigitLength])
#         # decPart.append(int(intDec[1][0, 4]))
#         decSeg.append(int(intDec[1][int(i*segDigitLength):int((i+1)*segDigitLength)]))
#     #   int(intDec[1])
#     #   print(intPart, decSeg)

#     intSeg = segmenting.segment(intPart, segNumInt, segDigitLength)
#     #   decSeg = segmenting.segment(decPart, segNumDec, segDigitLength)
#     if negSign:
#         for i in range(segNumInt):
#         intSeg[i] = -intSeg[i]
#         for i in range(segNumDec):
#         decSeg[i] = -decSeg[i]
#     return (intSeg, decSeg)
    
