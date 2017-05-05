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
    """
    表形式出力用のラベルを取得する.

    Returns
    -------
    result : str
        表形式出力用のラベル.

    """
    return "Time,Ground Acceleration,Absolute Acceleration,Relative Velocity,Relative Displacement"


  def result(self):
    """
    表形式で出力するための解析結果の配列を取得する.

    Returns
    -------
    result : ndarray
        解析結果の配列.
        列は時刻,地震加速度,絶対加速度,相対速度,相対変位の順になっている.

    """
    return np.c_[self.time, self.__acc_grd, self.acc_abs, self.vel, self.disp]


  def integrate(self):
    """
    選択した数値積分手法により1自由度系の地震応答解析を行う.

    """
    if   self.__method == "newmark":
      self.__newmark_beta_integration()
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


  def __newmark_beta_integration(self):
    """
    Newmark-β法により1自由度系の地震応答解析を行う.

    Notes
    -----
    * 非増分系で実装している.

    References
    ----------
    .. [1] 柴田明徳: 最新耐震構造解析(第3版), 森北出版, pp.97-108, 2014. 
       [2] 久田俊明, 野口裕久: 非線形有限要素法の基礎と応用, 丸善, pp.261-265, 1996.

    """
    k_m     = self.__k / self.__m
    omega   = math.sqrt(k_m)
    h       = self.__h
    c_m     = 2.0 * h * omega
    dt      = self.__dt
    beta    = self.__param["beta"]
    gamma   = self.__param["gamma"]

    # 相対速度
    acc_rel = np.zeros_like(self.__acc_grd, dtype=np.float64)

    # 初期値
    self.disp[0]    = 0.0
    self.vel[0]     = - self.__acc_grd[0] * dt
    self.acc_abs[0] = - 2.0 * h * omega * self.__acc_grd[0] * dt
    acc_rel[0]      = self.acc_abs[0] - self.__acc_grd[0]

    # 時間積分
    for i in range(0, self.__acc_grd.size - 1):
      acc_rel[i+1]      = - ( self.__acc_grd[i+1] + c_m*( self.vel[i] + dt*(1.0-gamma)*acc_rel[i] )                       \
                                                  + k_m*( self.disp[i] + dt*self.vel[i] + dt*dt*(0.5-beta)*acc_rel[i] ) ) \
                          / ( 1.0 + c_m*dt*gamma + k_m*dt*dt*beta )
      self.acc_abs[i+1] = acc_rel[i+1] + self.__acc_grd[i+1]
      self.disp[i+1]    = self.disp[i] + dt*self.vel[i] + dt*dt*( (0.5-beta)*acc_rel[i] + beta*acc_rel[i+1] )
      self.vel[i+1]     = self.vel[i] + dt*( (1.0-gamma)*acc_rel[i] + gamma*acc_rel[i+1] )


  def __nigam_jennings_integration(self):
    """
    Nigam-Jennings法により1自由度系の地震応答解析を行う.

    Notes
    -----
    * 時間刻みΔtは変化しないと仮定している.

    References
    ----------
    .. [1] 大崎順彦: 新・地震動のスペクトル解析入門, 鹿島出版会, pp.129-133, 1994.

    """
    omega  = math.sqrt(self.__k / self.__m)
    h      = self.__h
    dt     = self.__dt

    h2     = h * h
    omega2 = omega * omega
    omega3 = omega * omega * omega
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
    self.vel[0]     = - self.__acc_grd[0] * dt
    self.acc_abs[0] = - 2.0 * h * omega * self.__acc_grd[0] * dt

    # 時間積分
    for i in range(0, self.__acc_grd.size - 1):
      self.disp[i+1]    = a11*self.disp[i]      + a12*self.vel[i] \
                        + b11*self.__acc_grd[i] + b12*self.__acc_grd[i+1]
      self.vel[i+1]     = a21*self.disp[i]      + a22*self.vel[i] \
                        + b21*self.__acc_grd[i] + b22*self.__acc_grd[i+1]
      self.acc_abs[i+1] = - 2.0*h*omega*self.vel[i+1] - omega2*self.disp[i+1]