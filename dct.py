#!/usr/bin/env python3

################################################################################
#
# Discrete Cosine Transform.
# Written by Huki, file inception on 2016-08-04.
#
# Encode 24-bit or 32-bit bitmaps with DCT (and compress the result).
# Use multiprocessing to utilize all CPU cores.
# Optimized to be reasonably fast!
#
################################################################################

import os, sys
import math
import struct
import array
import zlib
import time
import multiprocessing

Quant_50 = [
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
]
Cos_table = [
  [math.cos((2*i+1)*j * math.pi/16) for j in range(8)] for i in range(8)
]
Range_list = [(i,j) for i in range(8) for j in range(8)]
Root2_inv = 1 / math.sqrt(2)

def ComputeDCT(a,u,v):
  r = 0
  for i,j in Range_list:
    r += a[i][j] * Cos_table[i][u] * Cos_table[j][v]
  if u == 0: r *= Root2_inv
  if v == 0: r *= Root2_inv
  r *= 0.25
  return r

def InverseDCT(a,i,j):
  r = 0
  for u,v in Range_list:
    c = a[u][v] * Cos_table[i][u] * Cos_table[j][v]
    if u == 0: c *= Root2_inv
    if v == 0: c *= Root2_inv
    r += c
  r *= 0.25
  return round(r)

def Shift(c):
  c -= 128
  return c

def Unshift(c):
  c += 128
  if c > 255: c = 255
  elif c < 0: c = 0
  return c

def Encode(array):
  array = [[Shift(c) for c in row] for row in array]
  array = [[ComputeDCT(array,u,v) for v in range(8)] for u in range(8)]
  array = [[round(a/q) for a,q in zip(a,q)] for a,q in zip(array, Quant_50)]
  return array

def Decode(array):
  array = [[a*q for a,q in zip(a,q)] for a,q in zip(array, Quant_50)]
  array = [[InverseDCT(array,i,j) for j in range(8)] for i in range(8)]
  array = [[Unshift(c) for c in row] for row in array]
  return array

def EncodePool(row):
  return [Encode(a) for a in row]

def DecodePool(row):
  return [Decode(a) for a in row]

################################################################################

def Generate(array, width, height):
  if width % 8:
    for a in array:
      a += [0] * (8 - width%8)
  w = len(array[0])
  if height % 8:
    array += [[0]*w] * (8 - height%8)
  h = len(array)
  
  print("  -> new", w, "x", h, "block")
  array = [[[[array[m+i][n+j] for j in range(8)] for i in range(8)] 
      for n in range(0,w,8)] for m in range(0,h,8)]
  print("  ->   generated", len(array[0]), "x", len(array), "blocks")
  return array

def Merge(array, width, height):
  w, h = len(array[0]), len(array)
  array = [[array[m][i][n][j] for i in range(w) for j in range(8)] 
      for m in range(h) for n in range(8)]
  print("  -> merged", len(array[0]), "x", len(array), "block")
  
  if height % 8:
    array = array[:height]
  if width % 8:
    array = [a[:width] for a in array]
  return array

def Split(array, width, height, n):
  w, h = width*n, height
  if w % 4: w += 4 - w%4
  array = [array[i:i+w] for i in range(0,w*h,w)]
  array = [a[:width*n] for a in array]
  array = [[a[i::n] for a in array] for i in range(n)]
  array = [Generate(c, width, height) for c in array]
  return array

def Join(array, width, height):
  w, h = width, height
  array = [Merge(c, width, height) for c in array]
  array = [[a[i][j] for j in range(w) for a in array] for i in range(h)]
  array = [a + [0] * (4 - len(a)%4) if len(a)%4 else a for a in array]
  array = [a for row in array for a in row]
  return array

################################################################################

def EncodeFile(filename):
  with open(filename, "rb") as f:
    data = f.read()
  
  hsz, = struct.unpack('I', data[10:14])
  width, height = struct.unpack('II', data[18:26])
  depth, = struct.unpack('H', data[28:30])
  
  header = list(data[:hsz])
  pixels = list(data[hsz:])
  
  print("Bitmap info: ")
  print("  width:", width, "  height:", height)
  print("  depth:", depth)
  
  print("Splitting channels:")
  channels = Split(pixels, width, height, int(depth/8))
  
  np = multiprocessing.cpu_count() ** 2
  print("Using", np, "processes")
  
  print("Encoding channels:", len(channels))
  t1 = time.time()
  with multiprocessing.Pool(np) as p:
    channels = [p.map(EncodePool, c) for c in channels]
  t2 = time.time()
  print("  Time elapsed:", t2-t1)
  
  print("Joining blocks:")
  pixels = Join(channels, width, height)
  
  filename = os.path.splitext(filename)[0] + ".dct"
  with open(filename, "wb") as f:
    f.write(bytearray(header))
    f.write(zlib.compress(array.array('b', pixels), 9))

def DecodeFile(filename):
  with open(filename, "rb") as f:
    data = f.read()
  
  hsz, = struct.unpack('I', data[10:14])
  width, height = struct.unpack('II', data[18:26])
  depth, = struct.unpack('H', data[28:30])
  
  header = list(data[:hsz])
  pixels = list(array.array('b', zlib.decompress(data[hsz:])))
  
  print("Bitmap info: ")
  print("  width:", width, "  height:", height)
  print("  depth:", depth)
  
  print("Splitting channels:")
  channels = Split(pixels, width, height, int(depth/8))
  
  np = multiprocessing.cpu_count() ** 2
  print("Using", np, "processes")
  
  print("Decoding channels:", len(channels))
  t1 = time.time()
  with multiprocessing.Pool(np) as p:
    channels = [p.map(DecodePool, c) for c in channels]
  t2 = time.time()
  print("  Time elapsed:", t2-t1)
  
  print("Joining blocks:")
  pixels = Join(channels, width, height)
  
  filename = os.path.splitext(filename)[0] + "_dec.bmp"
  with open(filename, "wb") as f:
    f.write(bytearray(header))
    f.write(bytearray(pixels))

def RunTests(filename):
  print("---------------------")
  print(" Running File Encode ")
  print("---------------------")
  EncodeFile(filename)
  print("---------------------")
  print(" Running File Decode ")
  print("---------------------")
  filename = os.path.splitext(filename)[0] + ".dct"
  DecodeFile(filename)

def main():
  if len(sys.argv) > 2:
    if (sys.argv[1] == "-encode"):
      EncodeFile(sys.argv[2])
      return
    elif (sys.argv[1] == "-decode"):
      DecodeFile(sys.argv[2])
      return
    elif (sys.argv[1] == "-test"):
      RunTests(sys.argv[2])
      return
  
  print("  Usage:", sys.argv[0], "-[encode|decode|test] <filename>")
  return


################################################################################

# call the main function
if __name__ == '__main__':
  main()
