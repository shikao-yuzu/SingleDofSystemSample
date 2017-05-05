# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from SingleDofSystem import SingleDofSystem

if __name__ == "__main__":
  """
  Pythonによる1自由度系の地震応答解析のサンプルコード.

  """
  #test_dir = "test/nigam/"
  test_dir = "test/newmark/"

  acc_grd  = np.genfromtxt(test_dir+"acc_grd.txt", dtype=np.float64)
  settings = json.load(open(test_dir+"settings.json", "r"))

  sdof = SingleDofSystem(acc_grd, **settings)
  sdof.integrate()

  # ファイル出力
  np.savetxt(test_dir+"result.csv", sdof.result(), fmt="%.10e", delimiter=",",
             header=sdof.header(), comments="")

  # プロット出力
  plt.subplot(2, 2, 1)
  plt.plot(sdof.time, acc_grd)
  plt.xlabel("Time [sec]")
  plt.ylabel("Ground Acceleration")

  plt.subplot(2, 2, 2)
  plt.plot(sdof.time, sdof.acc_abs)
  plt.xlabel("Time [sec]")
  plt.ylabel("Absolute Acceleration")

  plt.subplot(2, 2, 3)
  plt.plot(sdof.time, sdof.vel)
  plt.xlabel("Time [sec]")
  plt.ylabel("Relative Velocity")

  plt.subplot(2, 2, 4)
  plt.plot(sdof.time, sdof.disp)
  plt.xlabel("Time [sec]")
  plt.ylabel("Relative Displacement")

  plt.tight_layout()
  plt.savefig(test_dir+"result.pdf")
