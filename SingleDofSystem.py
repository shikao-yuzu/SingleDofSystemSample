# -*- coding: utf-8 -*-

import math
import numpy as np

class SingleDofSystem:
  def __init__(self, acc_grd, **settings):
    self.__acc_grd = acc_grd
    self.__m       = settings["m"]
    self.__k       = settings["k"]
    self.__h       = settings["h"]
    self.__dt      = settings["dt"]
    self.__method  = settings["method"]
    self.__param   = settings["param"]

    self.acc_abs   = np.zeros_like(self.__acc_grd, dtype=np.float64)
    self.vel       = np.zeros_like(self.__acc_grd, dtype=np.float64)
    self.disp      = np.zeros_like(self.__acc_grd, dtype=np.float64)
    self.time      = np.arange(0.0, self.__dt * self.__acc_grd.size, self.__dt, dtype=np.float64)


  def header(self):
    return "Time,Ground Acceleration,Absolute Acceleration,Relative Velocity,Relative Displacement"


  def result(self):
    return np.c_[self.time, self.__acc_grd, self.acc_abs, self.vel, self.disp]


  def integrate(self):
    if   self.__method == "newmark":
      raise NotImplementedError("Newmark beta method is not implemented.")
    elif self.__method == "nigam":
      self.__nigam_jennings_integration()
    elif self.__method == "houbolt":
      raise NotImplementedError("Houbolt method is not implemented.")
    elif self.__method == "wilson":
      raise NotImplementedError("Wilson theta method is not implemented.")
    elif self.__method == "runge_kutta":
      raise NotImplementedError("Runge-Kutta method is not implemented.")
    else:
      raise NotImplementedError("Unsupport integration method.")


  def __nigam_jennings_integration(self):
    omega  = math.sqrt(self.__k / self.__m)
    h      = self.__h
    dt     = self.__dt

    h2     = h*h
    omega2 = omega*omega
    omega3 = omega*omega*omega
    omegad = math.sqrt(1.0 - h2) * omega

    exp    = math.exp(-h*omega*dt)
    cos    = math.cos(omegad*dt)
    sin    = math.sin(omegad*dt)

    a11 =   exp * ( cos + h*omega*sin/omegad )
    a12 =   exp * sin/omegad
    a21 = - exp * omega2*sin/omegad
    a22 =   exp * ( cos - h*omega*sin/omegad )
    b11 =   exp * ( (1.0/omega2 + (2.0*h)/(omega3*dt))*cos                       \
                  + (h/(omega*omegad) - (1.0 - 2.0*h2)/(omega2*omegad*dt))*sin ) \
          - 2.0*h/(omega3*dt)
    b12 =   exp * ( - (2.0*h)/(omega3*dt)*cos + (1.0 - 2.0*h2)/(omega2*omegad*dt)*sin ) \
          - 1.0/omega2 + 2.0*h/(omega3*dt)
    b21 =   exp * ( - 1.0/(omega2*dt)*cos - (h/(omega*omegad*dt) + 1.0/omegad)*sin ) \
          + 1.0/(omega2*dt)
    b22 =   exp * ( 1.0/(omega2*dt)*cos + (h/(omega*omegad*dt))*sin ) \
          - 1.0/(omega2*dt)

    # 初期値
    self.disp[0]    = 0.0
    self.vel[0]     = - self.__acc_grd[0]*dt
    self.acc_abs[0] = - 2.0 * h * omega * self.__acc_grd[0] * dt

    # 時間積分
    for i in range(0, self.__acc_grd.size - 1):
      self.disp[i+1] = a11*self.disp[i]      + a12*self.vel[i] \
                     + b11*self.__acc_grd[i] + b12*self.__acc_grd[i+1]
      self.vel[i+1]  = a21*self.disp[i]      + a22*self.vel[i] \
                     + b21*self.__acc_grd[i] + b22*self.__acc_grd[i+1]
      self.acc_abs[i+1] = - 2.0*h*omega*self.vel[i+1] - omega2*self.disp[i+1]