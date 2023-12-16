
# Segmenting a long input x
# into at least num segments, each of bit length length
def segment(x, num = 1, length = 1):
  segs = []
  mask = (1 << (length)) - 1
  # print(mask)
  while x > 0:
    segs.append(mask & x)
    x = x >> length
  while len(segs) < num:
    segs.append(0)
  # print("original input:", x, "segments:", segs)
  return segs

# Segmenting x which is the decimal part of some float
# into at least num segments, each of bit length length
# Should be aligned on the left side
def segmentDecimal(x, num, length):
  segs = []
  mask = (1 << (length)) - 1
  # print(mask)
  while x > 0:
    segs.append(mask & x)
    x = x >> length
  while len(segs) < num:
    segs.append(0)
  # print("original input:", x, "segments:", segs)
  return segs 


def combine(segments, length):
  sum = 0
  n = len(segments)
  for i in range(n):
    sum += segments[i] << (length * i)
    # print(sum)
  return sum


def segNum(x, segLenth):
  num = 0
  while x > 0:
    num = num + 1
    x = x >> segLenth
  return num

# segs = segment(100, 3)
# print(combine(segs, 3))
# print()
