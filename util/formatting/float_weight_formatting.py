# from email.mime import multipart
import numpy
import util.formatting.segmenting as segmenting

# Segmenting a long input x
# into at least num segments, each of length length
def toSegments(weight, segDigitLength, segNumInt, segNumDec):
  # wstr = str(weight) 
  wstr = numpy.format_float_positional(weight)
  negSign = wstr[0] == '-'
  if negSign:
    wstr = wstr[1:]
#   print("is weight negative?", negSign)

#   intDec = intDec.rstrip("0")
  intDec = wstr.split(".") # integer part and decimal part
  intPart = int(intDec[0])

  if len(intDec) == 1:
    intDec.append("0")
  decLength = len(intDec[1])
  padding = "0" * (segDigitLength * segNumDec - decLength)
  intDec[1] = intDec[1] + padding
  decSeg = []
  for i in range(segNumDec):
    # print(intDec[1])
    # print(i*segDigitLength, (i+1)*segDigitLength)
    # print(intDec[1][i*segDigitLength, (i+1)*segDigitLength])
    # decPart.append(int(intDec[1][0, 4]))
    decSeg.append(int(intDec[1][int(i*segDigitLength):int((i+1)*segDigitLength)]))
#   int(intDec[1])
#   print(intPart, decSeg)

  intSeg = segmenting.segment(intPart, segNumInt, segDigitLength)
#   decSeg = segmenting.segment(decPart, segNumDec, segDigitLength)
  if negSign:
    for i in range(segNumInt):
      intSeg[i] = -intSeg[i]
    for i in range(segNumDec):
      decSeg[i] = -decSeg[i]
  return (intSeg, decSeg)



def toWeights(intSeg, decSeg, segDigitLength, segNumInt, segNumDec):
  intPart = segmenting.combine(intSeg, segDigitLength)
  multiplier = 10 ** (-segDigitLength)
  temp = multiplier
  decPart = 0
  for i in range(segNumDec):
    decPart += temp * decSeg[i]
    temp *= multiplier
  
  return intPart + decPart
  



intPart, decPart = toSegments(-1.3428927, 4, 1, 4)  
print(intPart, decPart)
print(toWeights(intPart, decPart, 4, 1, 3))


intPart, decPart = toSegments(-1, 4, 1, 4)  
print(intPart, decPart)
print(toWeights(intPart, decPart, 4, 1, 3))