# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 22:52:27 2023

@author: hbrhu

# equation
M = c0 * lg(Pd) + c1 * lg(ER) + c2
c0 = 1.23
c1 = 1.38
c2 = 5.39

# filter
3 Hz low filter

P_wave arrival
"""

import os
import pickle
import numpy as np
import numpy.ma as ma
import tensorflow as tf
from tqdm import tqdm
from scipy import integrate, signal


class KuyukAllen():
    """
    input waveforms -> output Maginitude Estimation Based on Accurate Location
    """
    def __init__(self, num0=20, thre=0.5, sample_rate=100):
        
        super(KuyukAllen, self).__init__()
        self.num0 = num0
        self.thre = thre
        self.sample_rate = sample_rate
        self.coeff = [1.23, 1.38, 5.39]
        self.sos = signal.butter(2, 3, 'low', output='sos', fs=sample_rate)
        
    def opfile(self, x):
        """load file"""
        with open(x, 'rb') as f:
            dic = pickle.load(f)
        Qname, Depth, Mag = dic['Qname'], dic['Depth'], dic['Mag']
        Qlat, Qlon, names = dic['Qlat'], dic['Qlon'], dic['Stations']
        lat, lon, R, pt = dic['lat'], dic['lon'], dic['R'], dic['pt']
        Accf = np.array(dic['Accf'])
        D = np.array([Depth for _ in R])
        RD = np.sqrt(D**2 + R**2)
        P_S = np.array(dic['P-S'])
        Ptim = np.array(dic['Ptime'])
        snr = np.array(dic['snra'])
        # 0 pt, 1.lat, 2.lon, 3.RD, 4.P_S, 5.Ptim, 6.snr
        loc_info = np.array([pt, lat, lon, R, P_S, Ptim, snr])
        loc_info = np.transpose(loc_info, (1, 0))
        return [[Qname, Depth, Mag, Qlat, Qlon], names[:self.num0], loc_info[:self.num0], Accf[:self.num0]]

    def disfil(self, x):
        """integration & filter"""
        x = x[:,:,2]
        x = signal.detrend(x, axis=1)
        x -= np.mean(x[:, :100], axis=1, keepdims=True)
        x = integrate.cumtrapz(x, dx=1/self.sample_rate, axis=1)
        x = integrate.cumtrapz(x, dx=1/self.sample_rate, axis=1)
        x = signal.sosfilt(self.sos, x, axis=1)
        return x
    
    def disidx(self, x, t_max):
        """tidxlis t from 1 to 30 s"""
        PtTr = x[:, 5][:, np.newaxis] - 14.99 #P_lc - RT
        pt = x[:, 0][:, np.newaxis]
        PS = x[:, 4][:, np.newaxis]
        
        tidxl = []
        for t in range(1, t_max+1, 1):
            # time threshold P_wave length
            t_mat = np.min(pt) - pt + t
            tg_mk = np.greater(t_mat + PtTr, self.thre) + 0.
            tl_mk = np.greater(t_mat, self.thre) + 0.
            t_mat = t_mat * tl_mk * tg_mk
            
            # tidx
            tidx = np.less_equal(np.arange(-97, 3001)[np.newaxis,:], t_mat*100) + 0.
            tidx *= tg_mk
            tlmt = np.less_equal(np.arange(-97, 3001)[np.newaxis,:], 4*400) + 0.
            tidx *= tlmt
            
            # P_S idx
            PSidx = np.less_equal(np.arange(-97, 3001)[np.newaxis,:], PS*100) + 0.
            PSidx *= tidx
            tidxl.append(PSidx)
        return tidxl
    
    def Predict(self, x, mask, r):
        # peak displacement
        x = ma.array(x, mask=1-mask)
        x = ma.max(ma.abs(x), axis=1)
        Pd = ma.log10(x)
        
        # epicenter r
        r = ma.array(r, mask=1-np.max(mask, axis=1))
        r = ma.log10(r)
        Ms = self.coeff[0]*Pd + self.coeff[1]*r + self.coeff[2]
        Ms = ma.mean(Ms)
        return Ms
    
    def __call__(self, file, t_max=30):
        Q_info, names, S_info, Acc = self.opfile(file)
        Dis = self.disfil(Acc)
        PSidx = self.disidx(S_info, t_max)
        Mjs = []
        for mask in PSidx:
            Mjs.append(self.Predict(Dis, mask[:,], S_info[:,3]))
        Mjs = np.array(Mjs)
        TriNs = np.sum(np.max(np.array(PSidx), axis=2), axis=1)
        Qname, Depth, Mag, Qlat, Qlon = [_ for _ in Q_info]
        return Qname, Depth, Mag, Qlat, Qlon, Mjs, TriNs, S_info
    
    
# file = '../20210213230800_Accf_Type1'
# Baseline = KuyukAllen()
# Event, D, M, Qlt, Qln, PM, TNs = Baseline(file)