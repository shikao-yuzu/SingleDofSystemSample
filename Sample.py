# -*- coding: utf-8 -*-

import json
import numpy as np
import matplotlib.pyplot as plt
from SingleDofSystem import SingleDofSystem

if __name__ == "__main__":
  acc_grd  = np.genfromtxt("acc_grd.txt", dtype=np.float64)
  settings = json.load(open("settings.json", "r"))

  sdof = SingleDofSystem(acc_grd, **settings)
  sdof.integrate()

  np.savetxt("result.csv", sdof.result(), fmt="%.10e", delimiter=",",
             header=sdof.header(), comments="")