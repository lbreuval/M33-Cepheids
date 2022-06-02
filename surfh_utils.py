#!/usr/bin/env python3

import itertools as it
import sys
from functools import partial
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import scipy as sp
import udft
from astropy.io import fits
from scipy.signal import convolve as conv
from scipy.signal import convolve2d as conv2

from surfh_code import ifu, models
from surfh_code import smallmrs as mrs
from surfh_code import utils
from surfh_code.algorithms import vox_reconstruction

def fov():
   fig, ax1 = plt.subplots(1, 1)
   fov = mrs.ch1a.fov
   ax1.plot(
      [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
       [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
       "-",
       alpha=1,
       color="blue",
   )
   fov = mrs.ch2a.fov
   ax1.plot(
      [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
       [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
       "-",
       alpha=1,
       color="green",
   )
   fov = mrs.ch3a.fov
   ax1.plot(
      [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
       [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
       "-",
       alpha=1,
       color="yellow",
   )
   fov = mrs.ch4a.fov
   ax1.plot(
      [v.alpha for v in fov.vertices] + [fov.vertices[0].alpha],
       [v.beta for v in fov.vertices] + [fov.vertices[0].beta],
       "-",
       alpha=1,
       color="red",
   )
   ax1.invert_xaxis()
   plt.show()
