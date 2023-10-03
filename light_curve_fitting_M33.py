from astropy.coordinates import ICRS, Distance, Angle
from astropy import units as u
import numpy as np
from astropy.io import ascii
import aplpy
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
import uncertainties
from uncertainties import ufloat
from scipy.optimize import curve_fit
from astroquery.vizier import Vizier
import pickle
from astropy.table import Table
import pandas as pd
from matplotlib import gridspec
import os.path
import os
from scipy.stats import norm
import scipy.optimize
import time
from astropy.wcs import WCS
from astropy.io import fits
from astropy.visualization import interval
import matplotlib.image as img
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import WCS
import matplotlib.colors as matcol
import warnings
from uncertainties import unumpy as unp

verboseTime=time.time()

###########################################################################################################################################################
fcolors = ['red', 'gold', 'green', 'blue', 'magenta']
data_color = {'F160W': 'r', 'F475W': 'b', 'F814W': 'g'}
fit_color  = 'blue'
phi_fine = np.linspace(0., 1., 100) 
offset_g2V = 1.
offset_i2V = 0.
###                68.3 %   90 %      99 %
delta_chi2 = {'1': 1., '2': 2.71, '3':6.386}


###########################################################################################################################################################
filters = {'b':'B', 'v':'V', 'i':'I', 'g':'g', 'r':'r', 'j':'i', 'a':'G' }
fil_templ = {'F160W': 'H', 'F475W': 'g', 'F814W': 'j'}

root_dr3 = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/m33cep_dr3/'
root_grd = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/m33cep_ground/'
root_dat = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/'
root_LCs = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Figures_M33/LCs_cleaned/'

bins     = {'0':[0., 2.5], '1':[0., 0.9], '2':[0.9, 1.2], '3':[1.2, 1.5], '4':[1.5, 2.5]}
bins_I15 = {'1':[np.log10(1),np.log10(3)], '2':[np.log10(3),np.log10(5)], '3':[np.log10(5),np.log10(7)], '4':[np.log10(7),np.log10(9.5)], '5':[np.log10(9.5),np.log10(10.5)] , 
			'6':[np.log10(10.5),np.log10(13.5)], '7':[np.log10(13.5),np.log10(15.5)], '8':[np.log10(15.5),np.log10(20)], '9':[np.log10(20),np.log10(30)], '10':[np.log10(30),2.5]}

### Flux for Vega (zero-point) in erg/cm2/s/A.
F_vega   = {'V': 532.761e-11, 'I': 113.04e-11, 'H': 14.2481e-11}

### Load ground data (BVI, gri):
tab_grd = pd.read_csv(root_grd+'pm11.out', header=0, sep="\s+")
### Load DR3 Cepheids (9 rows):
tab_dr3 = pd.read_csv(root_dr3+'pm11dr3.out', header=0, sep="\s+")
### Load HST data from Meredith:
tab_mered = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/meredith_cepheids_unstacked.dat', header=0, sep="\s+")
### Load Fourier parameters for BVI, gri:
fp = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/fourier_params.dat', header=0, sep="\s+")
### Read second file Macri (with period uncertainties):
pm11b = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/pm11b.out', header=0, sep="\s+")
### Read Fourier modes to fit for each star (all HST ceps with "3" flag and sig(phase)<0.05):
fmodes_to_fit = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/fourier_modes_to_fit.dat', header=0, sep="\s+")
### Read output parameters:
output_params = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/output_params.dat', header=0, sep="\s+")
### Geometry correction from Stefano's file:
geocorr = ascii.read(root_dat + '/Geom_corr_Stefano/distance_ratio.dat')


### Read phased data from Meredith:
f = open('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/meredith_cepheids_phased.dpy', 'rb')   
HST = pickle.load(f, encoding='latin1')
f.close()		

### Read M33 data from Lucas:
f = open('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/PM11_M33.dpy', 'rb')   
M33 = pickle.load(f, encoding='latin1')
f.close()	

gold_sample     = [X[3:19] for X in os.listdir(root_LCs) if X.startswith('lc_') and ('g' in X) and ('i' in X)         and (M33[X[3:19]]['flag_pm11']=='3') and (M33[X[3:19]]['sigma_phase']<0.05) and (X[3:19] in HST.keys())]
silver_g_sample = [X[3:19] for X in os.listdir(root_LCs) if X.startswith('lc_') and ('g' in X) and ('i' not in X)     and (M33[X[3:19]]['flag_pm11']=='3') and (M33[X[3:19]]['sigma_phase']<0.05) and (X[3:19] in HST.keys())]
silver_i_sample = [X[3:19] for X in os.listdir(root_LCs) if X.startswith('lc_') and ('g' not in X) and ('i' in X)     and (M33[X[3:19]]['flag_pm11']=='3') and (M33[X[3:19]]['sigma_phase']<0.05) and (X[3:19] in HST.keys())]
bronze_sample   = [X[3:19] for X in os.listdir(root_LCs) if X.startswith('lc_') and (M33[X[3:19]]['flag_pm11']=='3') and (X[3:19] in HST.keys()) and (X[3:19] not in gold_sample) and (X[3:19] not in silver_g_sample) and (X[3:19] not in silver_i_sample)]

### Stars with only one epoch per filter (or several points but too close, which makes to LC fit unsuccessful and unprecise):         
one_pt_per_filter = ['01333107+3031435', '01341586+3046125', '01333433+3034270', '01335346+3035354', '01335644+3047141', '01340071+3049589', '01340084+3049551', '01340474+3049181', '01340613+3037339', '01340955+3045357', '01341694+3046399', '01341754+3038196', '01341955+3049375', '01335911+3049049', '01333021+3041325', '01340498+3049285', '01332304+3032164', '01332556+3034273', '01332746+3037078', '01332875+3031316', '01332958+3031087', '01334139+3047364', '01334843+3050192', '01334880+3030448', '01341190+3029475', '01341955+3049375', '01342347+3042043', '01342385+3048589', '01343182+3043050', '01343169+3043002']
bad_LC_fit = ['01332847+3032351', '01332877+3037534', '01334104+3043399', '01334159+3036095', '01334263+3033298', '01334306+3045014', '01334322+3032434', '01334362+3047100', '01334398+3045570', '01334462+3044151', '01334645+3047405', '01335099+3031564', '01335305+3045414', '01335558+3044484', '01335668+3048378', '01335747+3049072', '01335894+3045307', '01342861+3048207', '01341720+3047514', '01341215+3043528', '01340864+3037547', ]
outliers = one_pt_per_filter + bad_LC_fit
gold_sample_extended = ['01340120+3048131', '01333241+3031437', '01334104+3043399', '01334167+3043115', '01335067+3034459', '01335232+3046026', '01340058+3036306', '01341343+3043340', '01342241+3044080']

outl2 = '01334390+3032452'


###########################################################################################################################################################
###
def get_period_bin(star, HST_fil):

	### For F160W (equivalent to H), take Inno+15 templates:
	if HST_fil == 'F160W':
		for B in bins_I15:
			if (M33[star]['logP'] >= bins_I15[B][0]) and (M33[star]['logP'] < bins_I15[B][1]):
				BIN = B
	### For F475W and F814W (equivalent to V and I), take M33 CFHT templates:
	elif HST_fil in ['F475W', 'F814W']:
		for B in bins:
			if (M33[star]['logP'] >= bins[B][0]) and (M33[star]['logP'] < bins[B][1]):
				BIN = B
	return(BIN)

### Return first guess amplitudes (ptp) in 3 HST filters, derived from ground light curves:
def first_guess_amplitudes(star):
	### Initialize first guess amplitude to 1 in the 3 filters:
	amp_first_guess = {'F160W': 0.35, 'F475W': 1., 'F814W': 0.5}

	### (1) V-band: If star has good "g" LC:
	if (star in gold_sample) or (star in silver_g_sample):
		amp_first_guess.update({'F475W': M33[star]['gamp'][0]})
	### Else, if star has good "i" LC, use A(I)/A(V) = 0.58 -> A(V) = A(I)/0.58:
	elif (star in silver_i_sample):
		amp_first_guess.update({'F475W': M33[star]['iamp'][0]/0.58})

	### (2) I-band: If star has good "i" LC:
	if (star in gold_sample) or (star in silver_i_sample):
		amp_first_guess.update({'F814W': M33[star]['iamp'][0]})
	### Else, if star has good "g" LC, use A(I)/A(V) = 0.58:
	elif (star in silver_g_sample):
		amp_first_guess.update({'F814W': 0.58*M33[star]['gamp'][0]})

	### (3) H-band: For F160W, use amplitude ratio from Inno+15: A(H)/A(V) = 0.34 (P<20) or 0.40 (P>20):
	if M33[star]['period'] <= 20:
		amp_first_guess.update({'F160W': 0.34*M33[star]['gamp'][0]})
	elif M33[star]['period'] > 20:
		amp_first_guess.update({'F160W': 0.40*M33[star]['gamp'][0]})

	return(amp_first_guess)

### Get I amplitude from V amplitude (relation from Yoachim+2009):
def ampI_from_ampV(ampV):
	return(0.58*ampV)

### Get H amplitude from V amplitude and period (relation from Inno+2015):
def aH_from_aV(ampV, per):
	if per <= 20:
		ampH = 0.34*ampV
	elif per > 20:
		ampH = 0.40*ampV
	return(ampH)

###
def Ah_over_Av(star):
	if M33[star]['period'] <= 20:
		ratio = 0.34
	elif M33[star]['period'] > 20:
		ratio = 0.40
	return(ratio)

### Return first guess reference phase in each band (mean along rising branch):
def first_guess_phases_MARB(star):
	### Initialize first guess phases to 0 in the 3 filters:
	phi_first_guess = {'F160W': 0, 'F475W': 0, 'F814W': 0}

	### In V, take it from the "g" light curve:
	JD_MARB_V, phi_MARB_V = JD_mean_along_rising_branch(star, 'g')
	phi_first_guess.update({'F475W': phi_MARB_V})
	
	### In I, take it from the "j" light curve:
	JD_MARB_I, phi_MARB_I = JD_mean_along_rising_branch(star, 'j')
	if ((JD_MARB_I, phi_MARB_I) == (0, 0)) and ((JD_MARB_V, phi_MARB_V) != (0, 0)):
		phi_MARB_I = phi_MARB_V + 0.02
	phi_first_guess.update({'F814W': phi_MARB_I})

	### In H, use phase-lag relation from Inno+2015:
	if ((JD_MARB_V, phi_MARB_V) != (0, 0)):
		phi_MARB_H = phi_MARB_V -0.002*M33[star]['logP'] + 0.08
	elif ((JD_MARB_I, phi_MARB_I) != (0, 0)):
		phi_MARB_H = phi_MARB_I -0.002*M33[star]['logP'] + 0.06
	phi_first_guess.update({'F160W': phi_MARB_H})
	return(phi_first_guess)

###
def color_title(star):
	if star in outliers:
		color = 'red'
	else:
		color = 'k'
	return(color)

###
def fourier_func(x, dPHI, Afact, A0, A1, A2, A3, A4, A5, A6, A7, P1, P2, P3, P4, P5, P6, P7):
	y = A0 \
	+ Afact*A1*np.cos(float(1)*2*np.pi*np.array(x+dPHI) + P1) \
	+ Afact*A2*np.cos(float(2)*2*np.pi*np.array(x+dPHI) + P2) \
	+ Afact*A3*np.cos(float(3)*2*np.pi*np.array(x+dPHI) + P3) \
	+ Afact*A4*np.cos(float(4)*2*np.pi*np.array(x+dPHI) + P4) \
	+ Afact*A5*np.cos(float(5)*2*np.pi*np.array(x+dPHI) + P5) \
	+ Afact*A6*np.cos(float(6)*2*np.pi*np.array(x+dPHI) + P6) \
	+ Afact*A7*np.cos(float(7)*2*np.pi*np.array(x+dPHI) + P7) 
	return(y)

### Function used to fit the light-curves (with additonal parameter d_PHI used to shift phase):
def fourier2(x, params):
	phi = [k for k in list(params.keys()) if k[:3]=='PHI']
	res = np.copy(x)
	res *=0
	for f in phi:
		res += params['dA']*params['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*np.array(x+params['d_PHI'])/params['WAV']+ params[f])
	if 'A0' in params:
		res += params['A0']
	return res

### Function used to fit the light-curves:
def leastsqFit(func, x, params, y, err=None, fitOnly=None, verbose=False, doNotFit=[], epsfcn=1e-7, ftol=1e-5, fullOutput=True, normalizedUncer=True, follow=None, maxfev=200, showBest=True):
	global Ncalls
	if fitOnly is None:
		if len(doNotFit)>0:
			fitOnly = [x for x in list(params.keys()) if x not in doNotFit]
		else:
			fitOnly = list(params.keys())
		fitOnly.sort() # makes some display nicer
	pfit = [params[k] for k in fitOnly]
	pfix = {}
	for k in list(params.keys()):
		if k not in fitOnly:
			pfix[k]=params[k]
	Ncalls=0
	plsq, cov, info, mesg, ier = \
			  scipy.optimize.leastsq(_fitFunc, pfit, args=(fitOnly,x,y,err,func,pfix,verbose,follow,), full_output=True, epsfcn=epsfcn, ftol=ftol, maxfev=maxfev)
	if cov is None:
		cov = np.zeros((len(fitOnly), len(fitOnly)))
	for i,k in enumerate(fitOnly):
		pfix[k] = plsq[i]

	xmodel = np.arange(0,1,0.001)
	model = func(xmodel,pfix)
	tmp = _fitFunc(plsq, fitOnly, x, y, err, func, pfix)
	try:
		chi2 = (np.array(tmp)**2).sum()
	except:
		chi2=0.0
		for x in tmp:
			chi2+=np.sum(x**2)
	reducedChi2 = chi2/float(np.sum([1 if np.isscalar(i) else
									 len(i) for i in tmp])-len(pfit)+1)
	if not np.isscalar(reducedChi2):
		reducedChi2 = np.mean(reducedChi2)
	uncer = {}
	for k in list(pfix.keys()):
		if not k in fitOnly:
			uncer[k]=0 # not fitted, uncertatinties to 0
		else:
			i = fitOnly.index(k)
			if cov is None:
				uncer[k]= -1
			else:
				uncer[k]= np.sqrt(np.abs(np.diag(cov)[i]))

				if normalizedUncer:
					uncer[k] *= np.sqrt(reducedChi2)
	if fullOutput:
		if normalizedUncer:
			try:
				cov *= reducedChi2
			except:
				pass
		cor = np.sqrt(np.diag(cov))
		cor = cor[:,None]*cor[None,:]
		cor = cov/cor
		pfix={'best':pfix, 'uncer':uncer, 'chi2':chi2, 'model':model, 'cov':cov, 'fitOnly':fitOnly, 'info':info, 'cor':cor, 'x':x, 'y':y, 'err':err, 'func':func}
	return pfix

### Function used to fit the light-curves:
def _fitFunc(pfit, pfitKeys, x, y, err=None, func=None, pfix=None, verbose=False, follow=None):
	global verboseTime, Ncalls
	Ncalls+=1
	params = {}
	for i,k in enumerate(pfitKeys):
		params[k]=pfit[i]
	for k in pfix:
		params[k]=pfix[k]
	if err is None:
		err = np.ones(np.array(y).shape)
	if type(y)==np.ndarray and type(err)==np.ndarray:
		if len(err.shape)==2:
			tmp = func(x,params)
			res = np.dot(np.dot(tmp-y, err), tmp-y)
			res = np.ones(len(y))*np.sqrt(res/len(y))
		else:
			y = np.array(y)
			res= ((func(x,params)-y)/err).flatten()
	else:
		res = []
		tmp = func(x,params)
		if np.isscalar(err):
			err = 0*y + err
		for k in range(len(y)):
			df = (np.array(tmp[k])-np.array(y[k]))/np.array(err[k])
			try:
				res.extend(list(df))
			except:
				res.append(df)
	if verbose and time.time()>(verboseTime+5):
		verboseTime = time.time()
		try:
			chi2=(res**2).sum/(len(res)-len(pfit)+1.0)
		except:
			chi2 = 0
			N = 0
			res2 = []
			for r in res:
				if np.isscalar(r):
					chi2 += r**2
					N+=1
					res2.append(r)
				else:
					chi2 += np.sum(np.array(r)**2)
					N+=len(r)
					res2.extend(list(r))
			res = res2
	return res

### Function used to fit the light-curves:
def randomParam(fit, N=None, x=None):
	if N is None:
		N = len(fit['x'])
	m = np.array([fit['best'][k] for k in fit['fitOnly']])
	res = [] # list of dictionnaries
	for k in range(N):
		p = dict(list(zip(fit['fitOnly'],np.random.multivariate_normal(m, fit['cov']))))
		p.update({k:fit['best'][k] for k in list(fit['best'].keys()) if not k in
				 fit['fitOnly']})
		res.append(p)
	ymin, ymax = None, None
	tmp = []
	if x is None:
		x = fit['x']
	for r in res:
		tmp.append(fit['func'](x, r))
	tmp = np.array(tmp)
	fit['r_param'] = res
	fit['r_ym1s'] = np.percentile(tmp, 16, axis=0)
	fit['r_yp1s'] = np.percentile(tmp, 84, axis=0)
	fit['r_x'] = x
	fit['r_y'] = fit['func'](x, fit['best'])
	fit['all_y'] = tmp
	return fit
randomParam = randomParam

### Run this function to get basic parameters (609 rows from PM11):
def make_M33_dict():

	### Create name for LC
	CID = [str(tab_grd['CID'][i]) for i in range(len(tab_grd))]
	FL = [str(tab_grd['FL'][i]) for i in range(len(tab_grd))]
	for i in range(len(CID)):
		if len(CID[i]) == 2:
			CID[i] = '00000'+CID[i]
		elif len(CID[i]) == 3:
			CID[i] = '0000'+CID[i]
		elif len(CID[i]) == 4:
			CID[i] = '000'+CID[i]
		elif len(CID[i]) == 5:
			CID[i] = '00'+CID[i]
		elif len(CID[i]) == 6:
			CID[i] = '0'+CID[i]
	for j in range(len(FL)):
		if len(FL[j]) == 1:
			FL[j] = '0'+FL[j]

	### Create dictionnary:
	M33 = {}
	for i in range(len(tab_grd)):
		M33[tab_grd['#M33SSS_ID'][i]] = {'lc_ID': '%s_%s'%(FL[i], CID[i]), 'flag_pm11': str(tab_grd['T'][i]), 
		'RA': tab_grd['RA'][i], 'DEC': tab_grd['Dec'][i],
		'period': tab_grd['P'][i], 'logP': np.log10(tab_grd['P'][i]), 'e_period': tab_grd['eP'][i],
		'MJD0': [50000+tab_grd['T_0'][i], tab_grd['eT0'][i] ], 'phmax': tab_grd['phmax'][i],
		'Bmag': [tab_grd['B'][i], tab_grd['eB'][i]], 'Bamp': [tab_grd['AB'][i], tab_grd['eAB'][i]],
		'Vmag': [tab_grd['V'][i], tab_grd['eV'][i]], 'Vamp': [tab_grd['AV'][i], tab_grd['eAV'][i]],
		'Imag': [tab_grd['I'][i], tab_grd['eI'][i]], 'Iamp': [tab_grd['AI'][i], tab_grd['eAI'][i]],
		'gmag': [tab_grd['g'][i], tab_grd['eg'][i]], 'gamp': [tab_grd['Ag'][i], tab_grd['eAg'][i]],
		'rmag': [tab_grd['r'][i], tab_grd['er'][i]], 'ramp': [tab_grd['Ar'][i], tab_grd['eAr'][i]],
		'imag': [tab_grd['i'][i], tab_grd['ei'][i]], 'iamp': [tab_grd['Ai'][i], tab_grd['eAi'][i]],
		'Gmag': [0., 0.], 'Gamp': [0., 0.]}

	### Add Gaia data to dictionnary:
	for k in range(len(tab_dr3)):
		M33[tab_dr3['#M33SSS_ID'][k]].update({'flag_pm11': str(tab_dr3['T'][k]), 
		'period': tab_dr3['P'][k], 'logP': np.log10(tab_dr3['P'][k]), 'e_period': tab_dr3['eP'][k],
		'MJD0': [50000+tab_dr3['T_0'][k], tab_dr3['eT0'][k] ], 'phmax': tab_dr3['phmax'][k],
		'Bmag': [tab_dr3['B'][k], tab_dr3['eB'][k]], 'Bamp': [tab_dr3['AB'][k], tab_dr3['eAB'][k]],
		'Vmag': [tab_dr3['V'][k], tab_dr3['eV'][k]], 'Vamp': [tab_dr3['AV'][k], tab_dr3['eAV'][k]],
		'Imag': [tab_dr3['I'][k], tab_dr3['eI'][k]], 'Iamp': [tab_dr3['AI'][k], tab_dr3['eAI'][k]],
		'gmag': [tab_dr3['g'][k], tab_dr3['eg'][k]], 'gamp': [tab_dr3['Ag'][k], tab_dr3['eAg'][k]],
		'rmag': [tab_dr3['r'][k], tab_dr3['er'][k]], 'ramp': [tab_dr3['Ar'][k], tab_dr3['eAr'][k]],
		'imag': [tab_dr3['i'][k], tab_dr3['ei'][k]], 'iamp': [tab_dr3['Ai'][k], tab_dr3['eAi'][k]],
		'Gmag': [tab_dr3['G'][k], tab_dr3['eG'][k]], 'Gamp': [tab_dr3['AG'][k], tab_dr3['eAG'][k]]})

	### Creates a list of MJD for all observations and get the average:
	for CEP in M33.keys():
		M33[CEP].update({'mean_MJD_PHATTER': 58000})
		if CEP in HST.keys():
			mjd_total = []
			for B in ['F160W', 'F475W', 'F814W']:
				for j in range(len(HST[CEP][B])):
					mjd_total.append(HST[CEP][B][j][3])
			M33[CEP].update({'mean_MJD_PHATTER': np.mean(mjd_total)})

	### Properties of M33:
	d_M33 = Distance(840, unit=u.kpc)
	RA_M33, DEC_M33 = 23.4625*u.deg, 30.6602*u.deg
	PA = 22.5*u.deg
	inclin = Angle('57deg')

	### Add galactocentric distance:
	# glx_ctr = ICRS(ra=RA_M33, dec=DEC_M33)
	# for X in M33.keys():
	# 	coord = ICRS(ra=M33[X]['RA']*u.deg, dec=M33[X]['DEC']*u.deg)
	# 	sky_radius = glx_ctr.separation(coord)
	# 	avg_dec = 0.5 * (glx_ctr.dec + coord.dec).radian
	# 	x = (glx_ctr.ra - coord.ra) * np.cos(avg_dec)
	# 	y = glx_ctr.dec - coord.dec
	# 	### azimuthal angle from coord to glx  -- not completely happy with this
	# 	phi = PA - Angle('90d') + Angle(np.arctan(y.arcsec / x.arcsec), unit=u.rad)
	# 	### convert to coord. in rotated frame, where y-axis is galaxy major ax; have to convert to arcmin bc can't do sqrt(x^2+y^2) when x and y are angles:
	# 	xp = (sky_radius * np.cos(phi.radian)).arcmin
	# 	yp = (sky_radius * np.sin(phi.radian)).arcmin
	# 	### de-project
	# 	ypp = yp / np.cos(inclin.radian)
	# 	obj_radius = np.sqrt(xp ** 2 + ypp ** 2)  ### in arcmin
	# 	obj_dist = Distance(np.tan(Angle(obj_radius, unit=u.arcmin).radian) * d_M33, unit=d_M33.unit)

	# 	M33[X].update({'d_galac': obj_dist.value})

	### Add geometry correction from Stefano's file:
	for X in M33.keys():
		dmod = 0.
		d_kpc = 0.
		if X in list(geocorr['Name']):
			dmod  = geocorr['delta_mod'][list(geocorr['Name']).index(X)]

			theta_as = geocorr['dplane'][list(geocorr['Name']).index(X)]
			d_kpc = 840*np.pi*theta_as/(180*3600)
		M33[X].update({'delta_mag': dmod, 'd_kpc': d_kpc })

	### Error on phase:
	for h in range(len(pm11b)):
		star = pm11b['#M33SSS_ID'][h]
		sig_phi = pm11b['eP'][h]*(M33[star]['mean_MJD_PHATTER']-pm11b['T_max'][h])/pm11b['P'][h]
		M33[star].update({'sigma_phase': sig_phi })

	_dBfile_data = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/PM11_M33.dpy'
	f = open(_dBfile_data, 'wb')
	pickle.dump(M33, f)
	f.close()

	return(M33)

### Run this function to create DPY file with phased HST data from Meredith:
### Select "stacked=True" to stack observations from a given field together.
def make_HST_dict(stacked=True):

	list_headers = os.listdir(root_dat + 'headers_phatter/')
	err_max = {'F475W': 0.020, 'F814W': 0.018}

	### Make a list of Cepheids in Meredith's file:
	cep_hst = []
	for k in range(len(tab_mered)):
		if tab_mered['Star_ID'][k] not in cep_hst:
			cep_hst.append(tab_mered['Star_ID'][k])

	HST = {}
	for X in cep_hst:
		print('%s  (%i/%i)'%(X, cep_hst.index(X)+1, len(cep_hst)))
		### Initialize dictionnary:
		HST[X] = {'F160W':[], 'F475W':[], 'F814W':[]}
		### Cepheids to exclude:
		excl_index = []

		for fil in ['F160W', 'F475W', 'F814W']:
			print('%s'%fil)

			### Delete short exposures (large error mag):
			ind1 = [ii for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (tab_mered['Chip'][ii]==1)]
			ind2 = [ii for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (tab_mered['Chip'][ii]==2)]

			if fil == 'F475W':
				for ind in [ind1, ind2]:
					### If only one field observed:
					if (len(ind) == 5) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind])):
						excl_index.append(ind[0])
					### If two fields observed:
					if (len(ind) == 10) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind[0:5]])):
						excl_index.append(ind[0])
					if (len(ind) == 10) and (tab_mered['ERR'][ind[5]] == np.max([tab_mered['ERR'][k] for k in ind[5:10]])):
						excl_index.append(ind[5])
					### If three fields observed:
					if (len(ind) == 15) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind[0:5]])):
						excl_index.append(ind[0])
					if (len(ind) == 15) and (tab_mered['ERR'][ind[5]] == np.max([tab_mered['ERR'][k] for k in ind[5:10]])):
						excl_index.append(ind[5])
					if (len(ind) == 15) and (tab_mered['ERR'][ind[10]] == np.max([tab_mered['ERR'][k] for k in ind[10:15]])):
						excl_index.append(ind[10])
					### If unusual number of exposures:
					for ii in ind:
						if tab_mered['ERR'][ii] > err_max['F475W']:
							excl_index.append(ii)

			if fil == 'F814W':
				for ind in [ind1, ind2]:
					### If only one field observed:
					if (len(ind) == 4) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind])):
						excl_index.append(ind[0])
					### If two fields observed:
					if (len(ind) == 8) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind[0:4]])):
						excl_index.append(ind[0])
					if (len(ind) == 8) and (tab_mered['ERR'][ind[4]] == np.max([tab_mered['ERR'][k] for k in ind[4:8]])):
						excl_index.append(ind[4])
					### If three fields observed:
					if (len(ind) == 12) and (tab_mered['ERR'][ind[0]] == np.max([tab_mered['ERR'][k] for k in ind[0:4]])):
						excl_index.append(ind[0])
					if (len(ind) == 12) and (tab_mered['ERR'][ind[4]] == np.max([tab_mered['ERR'][k] for k in ind[4:8]])):
						excl_index.append(ind[4])
					if (len(ind) == 12) and (tab_mered['ERR'][ind[8]] == np.max([tab_mered['ERR'][k] for k in ind[8:12]])):
						excl_index.append(ind[8])
					### If unusual number of exposures:
					for ii in ind:
						if tab_mered['ERR'][ii] > err_max['F814W']:
							excl_index.append(ii)

			### If flagged by Meredith, add to deleted lines:
			for ind in [ind1, ind2]:
				for k in ind:
					if tab_mered['FLAG'][k] != 0:
						excl_index.append(k)

			### Read meredith's file for this cepheid and this filter:
			image_name = [tab_mered['Image'][ii] for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (ii not in excl_index)]
			chip       = [tab_mered['Chip'][ii]  for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (ii not in excl_index)]
			mag        = [tab_mered['VEGA'][ii]  for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (ii not in excl_index)]
			emag       = [tab_mered['ERR'][ii]   for ii in range(len(tab_mered)) if (tab_mered['Star_ID'][ii]==X) and (tab_mered['Filter'][ii]==fil) and (ii not in excl_index)]

			### Get date for each image name given by Meredith:
			MJD = []
			for ii in range(len(image_name)):
				### Find header in list of headers sent by Meredith to get DATE:
				exp_info_file = [list_headers[k] for k in range(len(list_headers)) if list_headers[k].startswith(image_name[ii]) and ('chip%s'%chip[ii] in list_headers[k])][0]
				### Open and read header:
				head = open(root_dat+'headers_phatter/%s'%exp_info_file, 'r')
				all_data = [line.strip() for line in head.readlines()]

				### Get MJD either from 'PHOTMODE' (F475W, F814W') or from 'ROUTTIME':
				line_date_photmode = [all_data[k] for k in range(len(all_data)) if all_data[k].startswith('PHOTMODE')]
				line_date_routtime = [all_data[k] for k in range(len(all_data)) if all_data[k].startswith('ROUTTIME')]
				if len(line_date_photmode) != 0:
					mjd = float(line_date_photmode[0].split('#')[1].replace("' / observation con", ""))
				elif len(line_date_routtime) != 0:
					mjd = float(line_date_routtime[0].split('=')[1].split('/')[0])
				MJD.append(mjd)

			### Get phase:
			ph = ((MJD - M33[X]['MJD0'][0])/M33[X]['period'])%1.0

			### Option "stacked == False": keep all cleaned data points individually (4 pts in F160W, 4-8 pts in F475W and F814W):
			if stacked==False:
				HST[X].update({'%s'%fil: [(ph[ii], mag[ii], emag[ii], MJD[ii]) for ii in range(len(image_name))]})

			### Option "stacked == True": gather cleaned data points by field of view (1 pt in F160W, 1-2 pts in F475W and F814W):
			elif stacked==True:
				group_1, group_2, group_3, group_4, group_5 = [], [], [], [], []
				### Open group 1 initialized with first point:
				group_1 = [(ph[0], mag[0], emag[0], MJD[0])]
				### check if p and p-1 are close in phase and add in the same group:
				for p in range(len(image_name))[1::]:
					if abs(MJD[p]-MJD[p-1]) < 0.03:
						group_1.append((ph[p], mag[p], emag[p], MJD[p]))
					### If next point has a too different phase, keep last index in memory and break loop:
					else:
						break
				### If there are more points to stack after first group, start Group 2: 
				if p < len(ph)-1:
					group_2 = [(ph[p], mag[p], emag[p], MJD[p])]
					for q in range(len(image_name))[p+1::]:
						if abs(MJD[q]-MJD[q-1]) < 0.03:
							group_2.append((ph[q], mag[q], emag[q], MJD[q]))
						else:
							break
					### Start group 3:
					if q < len(ph)-1:
						group_3 = [(ph[q], mag[q], emag[q], MJD[q])]
						for r in range(len(image_name))[q+1::]:
							if abs(ph[r]-ph[r-1]) < 0.03:
								group_3.append((ph[r], mag[r], emag[r], MJD[r]))
							else:
								break
						### Start group 4:
						if r < len(ph)-1:
							group_4 = [(ph[r], mag[r], emag[r], MJD[r])]
							for s in range(len(image_name))[r+1::]:
								if abs(ph[s]-ph[s-1]) < 0.03:
									group_4.append((ph[s], mag[s], emag[s], MJD[s]))
								else:
									break

							if s < len(ph)-1:
								group_5 = [(ph[s], mag[s], emag[s], MJD[s])]
								for t in range(len(image_name))[s+1::]:
									if abs(ph[t]-ph[t-1]) < 0.03:
										group_5.append((ph[t], mag[t], emag[t], MJD[t]))
									else:
										break

								if t < len(ph)-1:
									print('\n %s has too many points (%i) in %s, some of them were skipped...'%(X, len(ph), fil))

				stack = [] 
				for GROUP in [group_1, group_2, group_3, group_4, group_5]:
					if len(GROUP) != 0:
						### Average phase, MJD, mag and errors over the group:
						mean_mjd      = np.mean([GROUP[ii][3] for ii in range(len(GROUP))])
						mean_mag      = np.mean([GROUP[ii][1] for ii in range(len(GROUP))])
						mean_err_stat = np.mean([GROUP[ii][2] for ii in range(len(GROUP))])
						mean_err_dev  = np.std([GROUP[ii][1] for ii in range(len(GROUP))])
						mean_phase    = ((mean_mjd - M33[X]['MJD0'][0])/M33[X]['period'])%1.0
						### Add averaged values to the stack of data:
						stack.append((mean_phase, mean_mag, np.max([mean_err_stat, mean_err_dev]), mean_mjd))

						HST[X].update({'%s'%fil: [s for s in stack]})

	_dBfile_data = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/meredith_cepheids_phased.dpy'
	f = open(_dBfile_data, 'wb')
	pickle.dump(HST, f)
	f.close()
	
### Take as input: ('b', 'v', 'i', 'g', 'r', 'j'):
def JD_mean_along_rising_branch(X, fil):

	BAND = filters[fil]
	mean_mag = M33[X]['%smag'%BAND][0]

	try:
		### Get light curve fit:
		LC_fit = pd.read_csv(root_grd + '/lc_%s.mod_%s'%(M33[X]['lc_ID'], fil), header=None, sep="\s+")
		### Select rising branch:
		xRB, yRB, mjdRB = [], [], []
		for i in range(len(LC_fit)-1):
			if LC_fit[0][i+1] < LC_fit[0][i]:
				xRB.append(LC_fit[1][i])
				yRB.append(LC_fit[0][i])
				mjdRB.append(50000+LC_fit[2][i])

		phi_MARB, jd_MARB = 0, 0
		for j in range(len(xRB)-1):
			if (yRB[j] >= mean_mag) and (yRB[j+1] <= mean_mag):
				phi_MARB = np.mean([xRB[j], xRB[j+1]])
				jd_MARB  = np.mean([mjdRB[j], mjdRB[j+1]])
			else:
				pass
	except:
		phi_MARB, jd_MARB = 0, 0


	# plt.figure(figsize=(6, 4)) 

	# plt.plot([0., 1.], [mean_mag, mean_mag], '-', color='limegreen', linewidth=1.)
	# plt.plot([x for x in LC_fit[1]], [y for y in LC_fit[0]], '-', color='b', linewidth=1.2, alpha=1.)
	# plt.plot(xRB, yRB, 'o', color='r', markersize=0.8)

	# plt.plot(phi_MARB, mean_mag, 'o', color='orange', markersize=5)
	# plt.xlim(0., 1)	
	# plt.gca().invert_yaxis()
	# plt.title('%s'%X)
	# plt.ylabel('%s'%BAND)
	# plt.xlabel('$\phi$')
	# plt.show()

	return(jd_MARB, phi_MARB)

### Derives the shift between the H and the g templates:
def rephasing_correction_H2V(star):
	### Load index for H template:
	indH = [ii for ii in range(len(fp)) if (fp['filter'][ii] == 'H') and (str(fp['bin'][ii]) == get_period_bin(star, 'F160W'))][0]
	### Build template line with mean mag 1:
	templ_h = 1.
	for f in range(1,8):
		templ_h += fp['A%i'%f][indH]*np.cos(float(f)*2*np.pi*np.array(phi_fine) + fp['PHI%i'%f][indH])

	### Load index for V template:
	indV = [ii for ii in range(len(fp)) if (fp['filter'][ii] == 'g') and (str(fp['bin'][ii]) == get_period_bin(star, 'F475W'))][0]
	### Build template line with mean mag 1:
	templ_g = 1.
	for i in range(1,8):
		templ_g += fp['A%i'%i][indV]*np.cos(float(i)*2*np.pi*np.array(phi_fine) + fp['PHI%i'%i][indV])

	### Find the phase at maximum of H-template and g-template:
	phase_maxH = phi_fine[list(templ_h).index(max(templ_h))]
	phase_maxg = phi_fine[list(templ_g).index(max(templ_g))]

	return(phase_maxH-phase_maxg-0.08)

###
def show_all_light_curves(star, fmode=[7, 7, 7], show=False, save=False):

	fm = {'F160W':fmode[0], 'F475W':fmode[1], 'F814W':fmode[2]}

	ftcol, flag_txt = 'k', ''
	if M33[star]['flag_pm11'] != '3':
		ftcol, flag_txt = 'r', '_F'

	plt.figure(figsize=(13, 8)) 
	gs=gridspec.GridSpec(3, 3, width_ratios=[1, 1, 1], height_ratios=[1, 1, 1])
	gs.update(wspace=0.25, hspace=0.5, left=0.06, bottom=0.07, right=0.98, top=0.94)

	ax0, ax1, ax2 = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])
	ax3, ax4, ax5 = plt.subplot(gs[3]), plt.subplot(gs[4]), plt.subplot(gs[5])
	ax6, ax7, ax8 = plt.subplot(gs[6]), plt.subplot(gs[7]), plt.subplot(gs[8])
	# ax9 = plt.subplot(gs[10])
	axes = {'F160W':ax0, 'F475W':ax1, 'F814W':ax2, 'B': ax3, 'V': ax4, 'I': ax5, 'g': ax6, 'r': ax7, 'i': ax8}# 'G': ax9}

	### Search for GROUND data:
	list_lc_grd = os.listdir(root_grd)
	for fil in list(filters.keys())[0:-1]:
		try:
			jd_MARB, phi_MARB = JD_mean_along_rising_branch(star, fil)
		except:
			jd_MARB, phi_MARB = 0., 0.

		BAND = filters[fil]
		ax = axes[BAND]
		X_dat, Y_dat, eY_dat, X_mod, Y_mod = [], [], [], [], []
		### Read data:
		if 'lc_%s.dto_%s'%(M33[star]['lc_ID'], fil) in list_lc_grd:
			data = ascii.read(root_grd+'lc_%s.dto_%s'%(M33[star]['lc_ID'], fil))
			X_dat  = list(data['col4'])
			Y_dat  = list(data['col2'])
			eY_dat = list(data['col3'])
			ax.errorbar(X_dat, Y_dat, yerr=eY_dat, fmt='o', color='k', markerfacecolor=data_color, alpha=0.5, markeredgewidth=0.4, markersize=5, capsize=0, ecolor=data_color, elinewidth=0.7)
		### Read model:
		if 'lc_%s.mod_%s'%(M33[star]['lc_ID'], fil) in list_lc_grd:
			model = ascii.read(root_grd+'lc_%s.mod_%s'%(M33[star]['lc_ID'], fil))
			X_mod = list(model['col2'])
			Y_mod = list(model['col1'])
			ampl = np.max(Y_mod)-np.min(Y_mod)
			ax.plot(X_mod, Y_mod, '-', color=fit_color, linewidth=1.2, alpha=1.)
			ax.annotate(filters[fil], xy=(0.04, np.max([Y_mod+[Y_dat[i]+eY_dat[i] for i in range(len(Y_dat))]])), xytext=(0.04, np.max([Y_mod+[Y_dat[i]+eY_dat[i] for i in range(len(Y_dat))]])), fontsize=11, weight='bold') 
		else:
			ax.annotate(filters[fil], xy=(0.04, 0.95), xytext=(0.04, 0.95), fontsize=11, weight='bold') 
		
		if M33[star]['%smag'%BAND][0] < 99.:
			ax.plot([0., 1.], [M33[star]['%smag'%BAND][0], M33[star]['%smag'%BAND][0]], '-', color='limegreen', linewidth=1.)
			ax.fill_between([0., 1.], [M33[star]['%smag'%BAND][0] + M33[star]['%smag'%BAND][1], M33[star]['%smag'%BAND][0] + M33[star]['%smag'%BAND][1]], [M33[star]['%smag'%BAND][0] - M33[star]['%smag'%BAND][1], M33[star]['%smag'%BAND][0] - M33[star]['%smag'%BAND][1]], facecolor='limegreen', alpha=0.25)
			ax.plot(phi_MARB, M33[star]['%smag'%BAND][0], 'o', color='orange', markersize=5)

		if filters[fil] in ['B', 'g', 'G']:
			ax.set_ylabel('mag', fontsize=12)
		if filters[fil] in ['g', 'r', 'i']:
			ax.set_xlabel('phase', fontsize=12)
		ax.set_xlim(0., 1)	
		ax.invert_yaxis()

	### Search for HST data:
	if star in HST.keys():
		for fil in ['F475W', 'F814W', 'F160W']:
			# print(' Performing fit in %s...'%fil)
			ax = axes[fil]
			data = HST[star][fil]
			X_dat  = [x[0] for x in data]
			Y_dat  = [x[1] for x in data]
			eY_dat = [x[2]*4 for x in data]
			ax.errorbar(X_dat, Y_dat, yerr=eY_dat, fmt='o', color='k', markerfacecolor=data_color, alpha=0.5, markeredgewidth=0.4, markersize=5, capsize=0, ecolor=data_color, elinewidth=0.7)
		
			### Sort the full data:
			Z = list(zip(X_dat, Y_dat))
			Z.sort()
			X_dat = np.array([X[0] for X in Z])
			Y_dat = np.array([X[1] for X in Z])

			ax.annotate(fil, xy=(0.035, np.mean(Y_dat)+0.45), xytext=(0.035, np.mean(Y_dat)+0.45), fontsize=9, weight='bold') 
			ax.set_xlim(0., 1)
			ax.invert_yaxis()

			fil_templ = fil_equiv_templ[fil]
			### First guess mean mag (A0) and first guess amplitude (A1):
			A0 = np.mean(Y_dat)

			### Get the exact template corresponding to the period bin:
			ind = [ii for ii in range(len(fp)) if (fp['filter'][ii] == fil_templ) and (str(fp['bin'][ii]) == get_period_bin(star, fil_templ))][0]

			### Fit in F475W:
			if fil in ['F475W', 'F814W']:
				A1 = M33[star]['%samp'%fil_templ][0]/2
				
			elif fil == 'F160W':
				### Get amplitude in V with two different methods:
				if (M33[star]['Vamp'][0] != 0.):
					Vamp = M33[star]['Vamp'][0]
				elif (M33[star]['Vamp'][0] == 0.):
					Vamp = (0.84-0.04*M33[star]['logP'])*M33[star]['gamp'][0]
				### Get amplitude expected in H:
				if (M33[star]['period'] <= 20):
					A1 = 0.34*Vamp/2
				elif (M33[star]['period'] > 20):
					A1 = 0.40*Vamp/2
				ax.set_ylabel('mag', fontsize=12)

			### Fourier params from template:
			fg0 = {'WAV':1., 'A0':A0, 'A1':A1, 'A2':fp['A2'][ind], 'PHI1':fp['PHI1'][ind], 'PHI2':fp['PHI2'][ind]}
			for p in range(3, fm[fil]+1):
				fg0.update({'A%i'%p: fp['A%s'%p][ind], 'PHI%i'%p: fp['PHI%s'%p][ind]})

			### Fits the light curve in magnitudes:
			# fit = leastsqFit(fourier, X_dat, fg0, Y_dat, err=np.ones(len(Y_dat)), verbose=2, doNotFit=['WAV', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7'])
			# fit = randomParam(fit)
			### Get output Fourier parameters:
			# fparams = fit['best']
			# print(fparams)

			# phi = [k for k in list(fparams.keys()) if k[:3]=='PHI']
			# fitline = fparams['A0']
			# for f in phi:
			# 	fitline += fparams['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*np.array(phi_fine)/fparams['WAV']+ fparams[f])
			# ax.plot(phi_fine, fitline, '-', color='m', linewidth=1.8)

			### Test: show successive light curves shifted in phase:
			# if fil == 'F475W':
			# 	phi = [k for k in list(fparams.keys()) if k[:3]=='PHI']
			# 	fitline_1 = fparams['A0']
			# 	for f in phi:
			# 		fitline_1 += fparams['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*(np.array(phi_fine)+0.05)/fparams['WAV']+ fparams[f])
			# 	ax.plot(phi_fine, fitline_1, '-', color='limegreen', linewidth=1.8)

			# 	fitline_2 = fparams['A0']
			# 	for f in phi:
			# 		fitline_2 += fparams['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*(np.array(phi_fine)+0.10)/fparams['WAV']+ fparams[f])
			# 	ax.plot(phi_fine, fitline_2, '-', color='deepskyblue', linewidth=1.8)

			# 	fitline_3 = fparams['A0']
			# 	for f in phi:
			# 		fitline_3 += fparams['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*(np.array(phi_fine)+0.15)/fparams['WAV']+ fparams[f])
			# 	ax.plot(phi_fine, fitline_3, '-', color='red', linewidth=1.8)

			# sigma = np.sqrt(sum( (np.array(ydata)-np.array(fitline))**2 )/len(xdata))

	ax1.set_title('%s  (P = %.3f d)'%(star, M33[star]['period']), fontsize=12, weight='bold', color=ftcol)
	if save==True:
		plt.savefig('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Figures_M33/LCs_HST_stacked/lc_%s.pdf'%(star))
	if show == True:
		plt.show()
	plt.close()

### Get phase difference between template and data:
def delta_phase_g(X):
    jd_MARB = JD_mean_along_rising_branch(X, 'g')[0]
    data = pd.read_csv(root_grd+'lc_%s.dto_g'%(M33[X]['lc_ID']), header=None, sep="\s+")
    phase_template = ((np.array(data[0])+50000-jd_MARB)/M33[X]['period'])%1.0
    phase_data = np.array(data[3])
    delta_phase = [phase_data[i]-phase_template[i] for i in range(len(data))]
    mean_delta_ph = np.mean([x for x in delta_phase if abs(x)<0.8])
    sigma_delta_ph = np.std([x for x in delta_phase if abs(x)<0.8])
    return(mean_delta_ph, sigma_delta_ph)

### Get phase difference between template and data:
def delta_phase_i(X):
    jd_MARB = JD_mean_along_rising_branch(X, 'j')[0]
    data = pd.read_csv(root_grd+'lc_%s.dto_j'%(M33[X]['lc_ID']), header=None, sep="\s+")
    phase_template = ((np.array(data[0])+50000-jd_MARB)/M33[X]['period'])%1.0
    phase_data = np.array(data[3])
    delta_phase = [phase_data[i]-phase_template[i] for i in range(len(data))]
    mean_delta_ph = np.mean([x for x in delta_phase if abs(x)<0.8])
    sigma_delta_ph = np.std([x for x in delta_phase if abs(x)<0.8])
    return(mean_delta_ph, sigma_delta_ph)

###########################################################################################################################################################
def fit_templates(star, precision='1', n_grid=10, show=True, save=True, show_template=False):
	### Uncomment next line to shut up warnings:
	warnings.filterwarnings("ignore")

	### Define phase lag between H and V:
	phlagHV = 0.08

	### Sets Fourier mode from file:
	ind_modes = list(fmodes_to_fit['cep']).index(star)
	fm = {'F160W':fmodes_to_fit['mode_H'][ind_modes], 'F475W':fmodes_to_fit['mode_V'][ind_modes], 'F814W':fmodes_to_fit['mode_I'][ind_modes]}
	# fm = {'F160W': 7, 'F475W': 7, 'F814W': 7}
	period, logP = M33[star]['period'], M33[star]['logP']

	print('\n (1) Checking data available for %s...'%star)
	if star in HST.keys():
		print('     -> %s has HST data   :) '%star)
		if (star in gold_sample):
			print('        + has "g" and "i" LCs  :D ')
		elif (star in silver_g_sample):
			print('        + has "g" LC only  :) ')
		elif (star in silver_i_sample):
			print('        + has "i" LC only  :) ')
		elif (star in bronze_sample):
			print('        + no "g" or "i" LC   :( ')
		if star in outliers:
			print('        !!! WARNING !!! This star is an outlier !!! ')
	else:
		print('\n %s has no HST data :('%star)
		return(0., 0., 0.)

	print('\n (2) Loading HST data for %s...'%star)
	dataH,   dataV,   dataI   =  HST[star]['F160W'],      HST[star]['F475W'],      HST[star]['F814W']
	X_datH,  X_datV,  X_datI  = [x[0]   for x in dataH], [x[0]   for x in dataV], [x[0]   for x in dataI]		### phase
	Y_datH,  Y_datV,  Y_datI  = [x[1]   for x in dataH], [x[1]   for x in dataV], [x[1]   for x in dataI]		### mag
	eY_datH, eY_datV, eY_datI = [x[2]*2 for x in dataH], [x[2]*2 for x in dataV], [x[2]*2 for x in dataI] 		### emag (multiplied by 2)
	### Exception:
	if star == '01334390+3032452':
		eY_datI = [x[2]*10 for x in dataI]
	print('     ->  F160W: %i   F475W: %i   F814W: %i '%(len(dataH), len(dataV), len(dataI)))

	print('\n (3) Setting first-guess mean magnitude (A0)...')
	A0_H, A0_V, A0_I = np.mean(Y_datH), np.mean(Y_datV), np.mean(Y_datI)
	print('     ->  A0(H) = %.3f   A0(V) = %.3f   A0(I) = %.3f '%(A0_H, A0_V, A0_I))

	print('\n (4) Phasing the V-template to the V light-curve...')
	if (star in gold_sample) or (star in silver_g_sample):
		(mean_delta_ph_g, sigma_delta_ph_g) = delta_phase_g(star) 			### mean shift in phase between template (g) and ground light curve
	elif (star in silver_i_sample):
		(mean_delta_ph_i, sigma_delta_ph_i) = delta_phase_i(star)
		mean_delta_ph_g, sigma_delta_ph_g = mean_delta_ph_i-0.027, sigma_delta_ph_i
	elif (star in bronze_sample):
		mean_delta_ph_g, sigma_delta_ph_g = 0., 0.
	print('     ->  Delta(phase) = %.4f ± %.4f'%(mean_delta_ph_g, sigma_delta_ph_g))

	print('\n (5) Setting first-guess V-band amplitude (A1)...')
	A1_V = 0.8
	if (star in gold_sample) or (star in silver_g_sample):
		A1_V = M33[star]['gamp'][0]
	elif (star in silver_i_sample):
		A1_V = M33[star]['iamp'][0]/0.58
	A1_V = A1_V/2  			### must be devided by two because input amplitude param. must be half ptp amplitude.
	print('     ->  A1(V) = %.3f'%(A1_V*2))

	### Getting index for the template...
	indH = [ii for ii in range(len(fp)) if (fp['filter'][ii] == 'H') and (str(fp['bin'][ii]) == get_period_bin(star, 'F160W'))][0]
	indV = [ii for ii in range(len(fp)) if (fp['filter'][ii] == 'g') and (str(fp['bin'][ii]) == get_period_bin(star, 'F475W'))][0]
	indI = [ii for ii in range(len(fp)) if (fp['filter'][ii] == 'j') and (str(fp['bin'][ii]) == get_period_bin(star, 'F814W'))][0]

	print('\n (6) Initializing grid-search...')
	av_grid  = np.linspace(max(0., A1_V*2-0.4), A1_V*2+0.4, n_grid)
	phv_grid = np.linspace(-0.05, 0.05, int(n_grid/2))

	### For bronze sample, extend the grid search in phase and in amplitude:
	if (which_sample(star)=='br') or (star in gold_sample_extended):
		av_grid  = np.linspace(max(0., A1_V*2-0.5), A1_V*2+0.5, n_grid)
		phv_grid = np.linspace(-0.50, 0.50, n_grid)
	print('     ->  %.2f < dA < %.2f   (%i steps)'%(min(av_grid), max(av_grid), len(av_grid)))
	print('     ->  %.2f < ph < %.2f   (%i steps)'%(min(phv_grid), max(phv_grid), len(phv_grid)))

	chis = np.zeros((len(av_grid) * len(phv_grid), 12))
	icnt = 0

	######################################################################################
	plt.figure(figsize=(10, 3.5)) 
	gs=gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
	gs.update(wspace=0.25, hspace=0.5, left=0.07, bottom=0.14, right=0.98, top=0.92)
	axH, axV, axI = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])

	### PLOT HST DATA POINTS:
	axH.errorbar(X_datH, Y_datH, yerr=eY_datH, fmt='o', color='k', markerfacecolor=data_color['F160W'], alpha=0.5, markeredgewidth=0.6, markersize=8, capsize=0, ecolor=data_color['F160W'], elinewidth=1., label='F160W')
	axV.errorbar(X_datV, Y_datV, yerr=eY_datV, fmt='o', color='k', markerfacecolor=data_color['F475W'], alpha=0.5, markeredgewidth=0.6, markersize=8, capsize=0, ecolor=data_color['F475W'], elinewidth=1., label='F475W')
	axI.errorbar(X_datI, Y_datI, yerr=eY_datI, fmt='o', color='k', markerfacecolor=data_color['F814W'], alpha=0.5, markeredgewidth=0.6, markersize=8, capsize=0, ecolor=data_color['F814W'], elinewidth=1., label='F814W')

	### GET AND PLOT THE "g" LIGHT CURVE to show reference phase (mean along rising branch):
	if (star in gold_sample) or (star in silver_g_sample): 
		data_g = ascii.read(root_grd+'lc_%s.dto_%s'%(M33[star]['lc_ID'], 'g'))
		X_dat_g, Y_dat_g, eY_dat_g = list(data_g['col4']), list(data_g['col2']+offset_g2V), list(data_g['col3'])
		model_g = ascii.read(root_grd+'lc_%s.mod_%s'%(M33[star]['lc_ID'], 'g'))
		X_mod_g, Y_mod_g = list(model_g['col2']), list(model_g['col1']+offset_g2V)
		# axV.plot(X_mod_g, Y_mod_g, '-', color='k', linewidth=1.2, alpha=1., label=r'$A_{\rm ptp}(g) = %.3f$'%(M33[star]['gamp'][0]))
		# axV.errorbar(X_dat_g, Y_dat_g, yerr=eY_dat_g, fmt='o', color='k', markerfacecolor='k', alpha=0.6, markeredgewidth=0.4, markersize=5, capsize=0, ecolor='k', elinewidth=0.7, label='g (CFHT)')
		# axV.plot([0., 1.], [M33[star]['gmag'][0]+offset_g2V, M33[star]['gmag'][0]+offset_g2V], '--', color='k', linewidth=1.)	
		# axV.plot(phi_MARB_V, M33[star]['gmag'][0]+offset_g2V, 'o', color='k', markerfacecolor='orange', alpha=0.9, markeredgewidth=0.4, markersize=8)

	### GET AND PLOT THE "i" LIGHT CURVE to show reference phase (mean along rising branch):
	if (star in gold_sample) or (star in silver_i_sample):
		data_i = ascii.read(root_grd+'lc_%s.dto_%s'%(M33[star]['lc_ID'], 'j'))
		X_dat_i, Y_dat_i, eY_dat_i = list(data_i['col4']), list(data_i['col2']+offset_i2V), list(data_i['col3'])
		model_i = ascii.read(root_grd+'lc_%s.mod_%s'%(M33[star]['lc_ID'], 'j'))
		X_mod_i, Y_mod_i = list(model_i['col2']), list(model_i['col1']+offset_i2V)
		# axI.plot(X_mod_i, Y_mod_i, '-', color='k', linewidth=1.2, alpha=1., label=r'$A_{\rm ptp}(i) = %.3f$'%(M33[star]['iamp'][0]))
		# axI.errorbar(X_dat_i, Y_dat_i, yerr=eY_dat_i, fmt='o', color='k', markerfacecolor='k', alpha=0.6, markeredgewidth=0.4, markersize=5, capsize=0, ecolor='k', elinewidth=0.7, label='i (CFHT)')
		# axI.plot([0., 1.], [M33[star]['imag'][0]+offset_i2V, M33[star]['imag'][0]+offset_i2V], '--', color='k', linewidth=1.)	
		# axI.plot(phi_MARB_I, M33[star]['imag'][0]+offset_i2V, 'o', color='k', markerfacecolor='orange', alpha=0.9, markeredgewidth=0.4, markersize=8)
	
	print('\n (7) Running grid-search...')
	for dA_trial in av_grid:
		# print('Grid-search %i / %i'%(list(av_grid).index(av_trial)+1, len(av_grid)))
		for phv_trial in phv_grid:
			### Set initial parameters in V from trial amplitude and phase, and derive corresponding trial parameters in I and H:
			fg_trial_V = {'WAV':1., 'dA':dA_trial,                  'A0':A0_V, 'A1':A1_V, 'PHI1':fp['PHI1'][indV], 'd_PHI': phv_trial-mean_delta_ph_g}
			fg_trial_I = {'WAV':1., 'dA':dA_trial*0.58            , 'A0':A0_I, 'A1':A1_V, 'PHI1':fp['PHI1'][indI], 'd_PHI': phv_trial-mean_delta_ph_g-0.027}
			fg_trial_H = {'WAV':1., 'dA':dA_trial*Ah_over_Av(star), 'A0':A0_H, 'A1':A1_V, 'PHI1':fp['PHI1'][indH], 'd_PHI': phv_trial-mean_delta_ph_g-phlagHV}
			### Add supplementary fourier modes:
			for p in range(2, fm['F475W']+1):
				fg_trial_V.update({'A%i'%p: fp['A%s'%p][indV], 'PHI%i'%p: fp['PHI%s'%p][indV]})
			for q in range(2, fm['F814W']+1):
				fg_trial_I.update({'A%i'%q: fp['A%s'%q][indI], 'PHI%i'%q: fp['PHI%s'%q][indI]})
			for r in range(2, fm['F160W']+1):
				fg_trial_H.update({'A%i'%r: fp['A%s'%r][indH], 'PHI%i'%r: fp['PHI%s'%r][indH]})
			### Fit the light curve with trial parameters:
			fitV = leastsqFit(fourier2, X_datV, fg_trial_V, Y_datV, err=eY_datV, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
			fitI = leastsqFit(fourier2, X_datI, fg_trial_I, Y_datI, err=eY_datI, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
			fitH = leastsqFit(fourier2, X_datH, fg_trial_H, Y_datH, err=eY_datH, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
			### Gather final parameters and derive chi2 from fit:
			fparamsV, fparamsI, fparamsH = fitV['best'], fitI['best'], fitH['best']
			chi2V, chi2I, chi2H = fitV['chi2'], fitI['chi2'], fitH['chi2']
			### The "0.06" number is used to smooth the chi2 value and give the term chi2(ampl) a consistent weight:
			if (which_sample(star)=='br') and (star not in gold_sample_extended):
				chi2_total = np.sum([chi2V, chi2I, chi2H])
			else:
				chi2_total = np.sum([chi2V, chi2I, chi2H]) + ((A1_V*2-dA_trial)**2 / 0.03**2)
			# print('dA = %.3f    dphi = %.2f   chi2: H=%.2f   V=%.2f   I=%.2f   Av=%.2f  ---> %.5f '%(dA_trial, phv_trial, chi2H, chi2V, chi2I, ((A1_V*2-dA_trial)**2 / 0.06**2), chi2_total))
			### Add output parameters of the fit to the results matrix:
			chis[icnt, 0] = phv_trial
			chis[icnt, 1] = dA_trial
			chis[icnt, 2] = np.mean(fitV['model'])
			chis[icnt, 3] = np.mean(fitI['model'])
			chis[icnt, 4] = np.mean(fitH['model'])
			chis[icnt, 5] = chi2_total
			chis[icnt, 6] = fitV['uncer']['A0']
			chis[icnt, 7] = fitI['uncer']['A0']
			chis[icnt, 8] = fitH['uncer']['A0']
			chis[icnt, 9]  = fitV['chi2']
			chis[icnt, 10] = fitI['chi2']
			chis[icnt, 11] = fitH['chi2']
			icnt = icnt+1

			fitlineV = fourier_func(phi_fine, phv_trial-mean_delta_ph_g,       dA_trial,                  fparamsV['A0'], A1_V, fp['A2'][indV], fp['A3'][indV], fp['A4'][indV], fp['A5'][indV], fp['A6'][indV], fp['A7'][indV], fp['PHI1'][indV], fp['PHI2'][indV], fp['PHI3'][indV], fp['PHI4'][indV], fp['PHI5'][indV], fp['PHI6'][indV], fp['PHI7'][indV])
			fitlineI = fourier_func(phi_fine, phv_trial-mean_delta_ph_g-0.027, dA_trial*0.58,             fparamsI['A0'], A1_V, fp['A2'][indI], fp['A3'][indI], fp['A4'][indI], fp['A5'][indI], fp['A6'][indI], fp['A7'][indI], fp['PHI1'][indI], fp['PHI2'][indI], fp['PHI3'][indI], fp['PHI4'][indI], fp['PHI5'][indI], fp['PHI6'][indI], fp['PHI7'][indI])
			fitlineH = fourier_func(phi_fine, phv_trial-mean_delta_ph_g-phlagHV, dA_trial*Ah_over_Av(star), fparamsH['A0'], A1_V, fp['A2'][indH], fp['A3'][indH], fp['A4'][indH], fp['A5'][indH], fp['A6'][indH], fp['A7'][indH], fp['PHI1'][indH], fp['PHI2'][indH], fp['PHI3'][indH], fp['PHI4'][indH], fp['PHI5'][indH], fp['PHI6'][indH], fp['PHI7'][indH])

			### Plot all lightcurves for each iteration:
			# axV.plot(phi_fine, fitlineV, '-', color=data_color['F475W'] ,linewidth=0.3)#, label='A = %.2f, phi = %.3f'%(av_trial, phv_trial))
			# axI.plot(phi_fine, fitlineI, '-', color=data_color['F814W'] ,linewidth=0.3)#, label='A = %.2f, phi = %.3f'%(av_trial, phv_trial))
			# axH.plot(phi_fine, fitlineH, '-', color=data_color['F160W'] ,linewidth=0.3)#, label='A = %.2f, phi = %.3f'%(av_trial, phv_trial))
	
	### Find the index of the iteration that minimized the chi2 -> final values:	
	idx = np.argmin(chis[:,5])
	pfinal, dAfinal, chi2final = chis[idx,0], chis[idx,1], chis[idx,5]
	print('     -> Final parameters: \n        ph = %.3f   dA = %.3f   (chi2_tot = %.2f)'%(pfinal, dAfinal, chi2final))

	### First guess of the fit = final parameters:
	fgV = {'WAV':1., 'dA':dAfinal,                  'A0':A0_V, 'A1':A1_V, 'PHI1':fp['PHI1'][indV], 'd_PHI': pfinal-mean_delta_ph_g}
	fgI = {'WAV':1., 'dA':dAfinal*0.58,             'A0':A0_I, 'A1':A1_V, 'PHI1':fp['PHI1'][indI], 'd_PHI': pfinal-mean_delta_ph_g-0.027}
	fgH = {'WAV':1., 'dA':dAfinal*Ah_over_Av(star), 'A0':A0_H, 'A1':A1_V, 'PHI1':fp['PHI1'][indH], 'd_PHI': pfinal-mean_delta_ph_g-phlagHV}
	### Add more modes to fit:
	for p in range(2, fm['F475W']+1):
		fgV.update({'A%i'%p: fp['A%s'%p][indV], 'PHI%i'%p: fp['PHI%s'%p][indV]})
	for q in range(2, fm['F814W']+1):
		fgI.update({'A%i'%q: fp['A%s'%q][indI], 'PHI%i'%q: fp['PHI%s'%q][indI]})
	for r in range(2, fm['F160W']+1):
		fgH.update({'A%i'%r: fp['A%s'%r][indH], 'PHI%i'%r: fp['PHI%s'%r][indH]})
	fitV_final = leastsqFit(fourier2, X_datV, fgV, Y_datV, err=eY_datV, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
	fitI_final = leastsqFit(fourier2, X_datI, fgI, Y_datI, err=eY_datI, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
	fitH_final = leastsqFit(fourier2, X_datH, fgH, Y_datH, err=eY_datH, verbose=2, doNotFit=['WAV', 'dA', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'PHI1', 'PHI2', 'PHI3', 'PHI4', 'PHI5', 'PHI6', 'PHI7', 'd_PHI'])
	fitV_final, fitI_final,fitH_final = randomParam(fitV_final), randomParam(fitI_final), randomParam(fitH_final)
	fparamsV_final, fparamsI_final, fparamsH_final = fitV_final['best'],   fitI_final['best'],   fitH_final['best']

	model_V = fourier_func(phi_fine, pfinal-mean_delta_ph_g,       dAfinal,                  fparamsV_final['A0'], A1_V, fp['A2'][indV], fp['A3'][indV], fp['A4'][indV], fp['A5'][indV], fp['A6'][indV], fp['A7'][indV], fp['PHI1'][indV], fp['PHI2'][indV], fp['PHI3'][indV], fp['PHI4'][indV], fp['PHI5'][indV], fp['PHI6'][indV], fp['PHI7'][indV])
	model_I = fourier_func(phi_fine, pfinal-mean_delta_ph_g-0.027, dAfinal*0.58,             fparamsI_final['A0'], A1_V, fp['A2'][indI], fp['A3'][indI], fp['A4'][indI], fp['A5'][indI], fp['A6'][indI], fp['A7'][indI], fp['PHI1'][indI], fp['PHI2'][indI], fp['PHI3'][indI], fp['PHI4'][indI], fp['PHI5'][indI], fp['PHI6'][indI], fp['PHI7'][indI])
	model_H = fourier_func(phi_fine, pfinal-mean_delta_ph_g-phlagHV, dAfinal*Ah_over_Av(star), fparamsH_final['A0'], A1_V, fp['A2'][indH], fp['A3'][indH], fp['A4'][indH], fp['A5'][indH], fp['A6'][indH], fp['A7'][indH], fp['PHI1'][indH], fp['PHI2'][indH], fp['PHI3'][indH], fp['PHI4'][indH], fp['PHI5'][indH], fp['PHI6'][indH], fp['PHI7'][indH])
	axV.plot(phi_fine, model_V, '-', color=data_color['F475W'], linewidth=1.8)#, label=r'A$_{\rm ptp}(V)$=%.3f'%(max(model_V)-min(model_V)))
	axI.plot(phi_fine, model_I, '-', color=data_color['F814W'], linewidth=1.8)#, label=r'A$_{\rm ptp}(I)$=%.3f'%(max(model_I)-min(model_I)))
	axH.plot(phi_fine, model_H, '-', color=data_color['F160W'], linewidth=1.8)#, label=r'A$_{\rm ptp}(H)$=%.3f'%(max(model_H)-min(model_H)))

	### Mean magnitudes from flux:
	mv = flux2mag(np.mean([mag2flux(y, 'V') for y in model_V]), 'V')
	mi = flux2mag(np.mean([mag2flux(y, 'I') for y in model_I]), 'I')
	mh = flux2mag(np.mean([mag2flux(y, 'H') for y in model_H]), 'H')
	e_mv_stat, e_mi_stat, e_mh_stat = chis[idx,6], chis[idx,7], chis[idx,8]

	### Same but for individual chi2s for each mean mag:
	mags_v_in = [chis[i,2] for i in range(len(chis)) if chis[i,9] < np.min([chis[i,9]  for i in range(len(chis))])+delta_chi2[precision]]
	mags_i_in = [chis[i,3] for i in range(len(chis)) if chis[i,10]< np.min([chis[i,10] for i in range(len(chis))])+delta_chi2[precision]]
	mags_h_in = [chis[i,4] for i in range(len(chis)) if chis[i,11]< np.min([chis[i,11] for i in range(len(chis))])+delta_chi2[precision]]
	e_mv_tot, e_mi_tot, e_mh_tot = 0.5*(max(mags_v_in)-min(mags_v_in)), 0.5*(max(mags_i_in)-min(mags_i_in)), 0.5*(max(mags_h_in)-min(mags_h_in))

	### Artificially increase uncertainties for a few stars whose fit returns an error of zero:
	if star == '01332768+3034238':
		e_mv_tot, e_mi_tot = 0.035, 0.035
	elif star == '01333015+3038039':
		e_mv_tot, e_mi_tot = 0.050, 0.050
	elif star == '01333039+3035555':
		e_mi_tot = 0.020
	elif star == '01334983+3037587':
		e_mv_tot, e_mi_tot = 0.020, 0.020
	elif star == '01335075+3035444':
		e_mv_tot, e_mi_tot = 0.020, 0.020
	elif star == '01335247+3038442':
		e_mv_tot, e_mi_tot = 0.080, 0.020
	elif star == '01341459+3044135':
		e_mv_tot, e_mi_tot = 0.080, 0.080
	elif star == '01334120+3035504':
		e_mv_tot = 0.080
	elif star == '01335646+3044420':   ###
		e_mi_tot = 0.105
	elif star == '01341259+3041262':
		e_mi_tot = 0.035
	elif star == '01334228+3037474':   ###
		e_mv_tot = 0.025
	elif star == '01341343+3043340':
		e_mv_tot = 0.085
	elif star == '01341383+3044184':
		e_mi_tot = 0.015
	elif star == '01341387+3043240':
		e_mi_tot = 0.040
	elif star == '01334228+3037474':
		e_mv_tot = 0.040
	elif star == '01333880+3037515':
		e_mv_tot, e_mi_tot = 0.025, 0.030
	elif star == '01334167+3043115':
		e_mi_tot = 0.040
	elif star == '01334821+3038001':
		e_mv_tot, e_mi_tot = 0.020, 0.025
	elif star == '01335886+3037198':
		e_mi_tot = 0.070
	elif star == '01335947+3032266':
		e_mv_tot = 0.035
	elif star == '01340817+3039318':
		e_mv_tot, e_mi_tot = 0.080, 0.070

	### PLOT MEAN MAGNITUDES FROM FLUX:
	axV.plot([0., 1.], [mv, mv], '--', color=data_color['F475W'], linewidth=1.5, label='$m = %.3f \pm %.3f$'%(mv, e_mv_tot))
	axI.plot([0., 1.], [mi, mi], '--', color=data_color['F814W'], linewidth=1.5, label='$m = %.3f \pm %.3f$'%(mi, e_mi_tot))
	axH.plot([0., 1.], [mh, mh], '--', color=data_color['F160W'], linewidth=1.5, label='$m = %.3f \pm %.3f$'%(mh, e_mh_tot))	

	axV.fill_between([0., 1.], [mv+e_mv_tot, mv+e_mv_tot], [mv-e_mv_tot, mv-e_mv_tot], facecolor=data_color['F475W'], alpha=0.15)
	axI.fill_between([0., 1.], [mi+e_mi_tot, mi+e_mi_tot], [mi-e_mi_tot, mi-e_mi_tot], facecolor=data_color['F814W'], alpha=0.15)
	axH.fill_between([0., 1.], [mh+e_mh_tot, mh+e_mh_tot], [mh-e_mh_tot, mh-e_mh_tot], facecolor=data_color['F160W'], alpha=0.15)
	
	### PLOT THE TEMPLATES ("g", "i", "H") with their initial parameters:
	if show_template == True:
		templ_g = fourier_func(phi_fine, -mean_delta_ph_g,       1, A0_V, fp['A1'][indV], fp['A2'][indV], fp['A3'][indV], fp['A4'][indV], fp['A5'][indV], fp['A6'][indV], fp['A7'][indV], fp['PHI1'][indV], fp['PHI2'][indV], fp['PHI3'][indV], fp['PHI4'][indV], fp['PHI5'][indV], fp['PHI6'][indV], fp['PHI7'][indV])
		templ_i = fourier_func(phi_fine, -mean_delta_ph_g-0.027, 1, A0_I, fp['A1'][indI], fp['A2'][indI], fp['A3'][indI], fp['A4'][indI], fp['A5'][indI], fp['A6'][indI], fp['A7'][indI], fp['PHI1'][indI], fp['PHI2'][indI], fp['PHI3'][indI], fp['PHI4'][indI], fp['PHI5'][indI], fp['PHI6'][indI], fp['PHI7'][indI])
		templ_h = fourier_func(phi_fine, -mean_delta_ph_g,       1, A0_H, fp['A1'][indH], fp['A2'][indH], fp['A3'][indH], fp['A4'][indH], fp['A5'][indH], fp['A6'][indH], fp['A7'][indH], fp['PHI1'][indH], fp['PHI2'][indH], fp['PHI3'][indH], fp['PHI4'][indH], fp['PHI5'][indH], fp['PHI6'][indH], fp['PHI7'][indH])
		axV.plot(phi_fine, templ_g, '-', color=data_color['F475W'], linewidth=3., alpha=0.15, label='Template ($g$)')
		axI.plot(phi_fine, templ_i, '-', color=data_color['F814W'], linewidth=3., alpha=0.15, label='Template ($i$)')	
		axH.plot(phi_fine, templ_h, '-', color=data_color['F160W'], linewidth=3., alpha=0.15, label='Template ($H$)')
	else:
		afinal = max(model_V)-min(model_V)
		axV.set_ylim(mv-(afinal/2+0.3), mv+(afinal/2+0.3))
		axI.set_ylim(mi-(afinal/2+0.3), mi+(afinal/2+0.3))
		axH.set_ylim(mh-(afinal/2+0.3), mh+(afinal/2+0.3))

	axH.set_xlim(0., 1)
	axV.set_xlim(0., 1)
	axI.set_xlim(0., 1)
	axH.invert_yaxis()
	axV.invert_yaxis()
	axI.invert_yaxis()
	axH.set_xlabel('phase', fontsize=12)
	axV.set_xlabel('phase', fontsize=12)
	axI.set_xlabel('phase', fontsize=12)
	axH.legend(loc='lower left', fontsize=10)
	axV.legend(loc='lower left', fontsize=10)
	axI.legend(loc='lower left', fontsize=10)
	axH.set_ylabel('mag', fontsize=12)
	axV.set_title('%s  (P = %.3f d)'%(star, M33[star]['period']), fontsize=12, weight='bold', color=color_title(star))
	if save==True:
		# plt.savefig('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Figures_M33/Final_HST_LC_fits/lc_%s.pdf'%(star))
		### Save output parameters to file:
		line_to_export = '%s 	%.4f   		%.3f    %.3f      %.3f    %.3f      %.3f    %.3f    %s      %.4f    %.4f    %.2f     \n'%(star, M33[star]['period'], mh, e_mh_tot, mv, e_mv_tot, mi, e_mi_tot, which_sample(star), pfinal, max(model_V)-min(model_V), chis[idx,5])
		replace_line(root_dat+'output_params.dat', list(output_params['cep']).index(star)+1, line_to_export)

	if show == True:
		plt.show()
	plt.close()

	print('\n (7) Final results: ')
	print('        mh = %.3f ± %.3f   mv = %.3f ± %.3f   mi = %.3f ± %.3f    '%(mh, e_mh_tot, mv, e_mv_tot, mi, e_mi_tot))
	mW = ufloat(mh, e_mh_tot) - 0.386*(0.065 + 0.658*(ufloat(mv, e_mv_tot)-ufloat(mi, e_mi_tot)))
	print('    --- mW = %.3f ± %.3f --- \n'%(mW.nominal_value, mW.std_dev))

	if star in outliers:
		print('        ########################################### ')
		print('        !!! WARNING !!! This star is an outlier !!! ')
		print('        -> Should NOT be considered in the PLR /!\  ')
		print('        ########################################### \n ')

	return(mh, mv, mi)


########################################################################################################################################################################
###
def plot_errors(fil):

	mags = [tab_mered['VEGA'][i] for i in range(len(tab_mered)) if tab_mered['Filter'][i] == fil and tab_mered['FLAG'][i] == 0]
	errs = [tab_mered['ERR'][i]  for i in range(len(tab_mered)) if tab_mered['Filter'][i] == fil and tab_mered['FLAG'][i] == 0]
	inds = [i                    for i in range(len(tab_mered)) if tab_mered['Filter'][i] == fil and tab_mered['FLAG'][i] == 0]

	if fil == 'F475W':
		err_max = 0.020
	elif fil == 'F814W':
		err_max = 0.018

	plt.figure(figsize=(14,6))
	plt.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.12, hspace=0.1, wspace=0.3)

	plt.plot([np.min(inds), np.max(inds)], [0., 0.], '-', linewidth=0.9, color='k')
	plt.plot(inds, errs, 'o', color='k', markerfacecolor='b', markeredgewidth=0.8, markersize=5)
	plt.plot([inds[i] for i in range(len(inds)) if errs[i] > err_max], [errs[i] for i in range(len(inds)) if errs[i] > err_max], 'o', color='k', markerfacecolor='r', markeredgewidth=0.8, markersize=5)

	plt.xlabel('Indices', fontsize=11)
	plt.ylabel('err', fontsize=11)
	plt.xlim(np.min(inds)-10, np.max(inds)+10)
	plt.ylim(0., 0.250)
	plt.title('%s'%fil, fontsize=10)
	plt.show()

### Takes as input filters: 
def plot_templates(fil):

	indices = [ii for ii in range(len(fp)) if (fp['filter'][ii] == fil)]

	for ind in indices:
		fg0 = {'A0':fp['A0'][ind], 'A1':fp['A1'][ind], 'A2':fp['A2'][ind], 'A3':fp['A3'][ind], 'A4':fp['A4'][ind], 'A5':fp['A5'][ind], 'A6':fp['A6'][ind], 'A7':fp['A7'][ind], 'PHI1':fp['PHI1'][ind], 
			'PHI2':fp['PHI2'][ind], 'PHI3':fp['PHI3'][ind], 'PHI4':fp['PHI4'][ind], 'PHI5':fp['PHI5'][ind], 'PHI6':fp['PHI6'][ind], 'PHI7':fp['PHI7'][ind], 'WAV':1.}

		phi = [k for k in list(fg0.keys()) if k[:3]=='PHI']
		fitline = fg0['A0']
		for f in phi:
			fitline += fg0['A'+f[3:]]*np.cos(float(f[3:])*2*np.pi*np.array(phi_fine)/fg0['WAV']+ fg0[f])

		plt.plot(phi_fine, fitline, '-', linewidth=1.8, label='Bin %s'%str(fp['bin'][ind]))

	plt.legend()
	plt.show()

### Fits templates to HST sample:
def template_fitting_all_PHATTER_ceps(n_grid, precision):

	fparams_file = root_dat + 'output_params.dat'
	first_line = 'cep                 period 	        H 	eH 	V 	eV 	I 	eI 	sample 	phase 	Av 	chi2	 \n'

	cep = list(fmodes_to_fit['cep'])

	### Erases all that was previously written in this file:
	with open(fparams_file,"w") as mf:
		mf.write(first_line)
		for X in cep:			
			mf.write('%s 	%.4f   		0.      0.      0.      0.      0.      0.      0.      %s 	   1.      0.      \n'%(X, M33[X]['period'], which_sample(X)))

	fparams_file = root_dat + 'output_params.dat'
	for X in cep:
		(mh, mv, mi) = fit_templates(X, precision=precision, n_grid=n_grid, show=False, save=True)
		print(' %s   (%i / %i)    mh=%.3f   mv=%.3f   mi=%.3f  \n ###################################################################'%(X, cep.index(X)+1, len(cep), mh, mv, mi))

###
def plot_sigma_phi():

	plt.figure(figsize=(6,5))
	plt.subplots_adjust(left=0.12, right=0.97, top=0.96, bottom=0.10, hspace=0.1, wspace=0.3)

	plt.plot([0., 2.5], [np.log10(0.05), np.log10(0.05)], '--', linewidth=0.9, color='k', alpha=0.6)
	plt.plot([M33[X]['logP'] for X in M33.keys()], [np.log10(M33[X]['sigma_phase']) for X in M33.keys()], 'o', color='k', markerfacecolor='b', markeredgewidth=0.8, markersize=5)

	plt.xlabel('$\log P$ (days)', fontsize=11)
	plt.ylabel('$\log \sigma(\phi)$', fontsize=11)
	plt.xlim(0.25, 2.25)
	# plt.ylim(0., 0.250)
	plt.show()

### Show spatial distribution of HST Cepheids and template Cepheids:
def show_spatial_distribution(fil):

	with fits.open('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Make_postage_stamps_pairs/data_pairs/m33_%s.fits'%fil) as hdu:

		w = WCS(hdu[0].header)
		plt.figure(figsize=(8,8.5))
		plt.subplot(projection=w)
		plt.subplots_adjust(top=0.94, bottom=0.08, left=0.18, right=0.97, hspace=0.2, wspace=0.2)

		scale = interval.ZScaleInterval(nsamples=600, contrast=0.25, max_reject=0.5, min_npixels=5, krej=2.5, max_iterations=5)
		(vmin, vmax) = scale.get_limits(hdu[0].data)
	   
		img_norm = matcol.Normalize(vmin=vmin, vmax=vmax)
		plt.imshow(hdu[0].data, cmap='gray', norm=img_norm)

		# plt.plot([M33[X]['RA'] for X in M33.keys() if M33[X]['flag_pm11'] == '3'], [M33[X]['DEC'] for X in M33.keys() if M33[X]['flag_pm11'] == '3'], 'o', color='r', mfc='none', markeredgewidth=1.0, markersize=6, label='Ground sample')
		# plt.plot([M33[X]['RA'] for X in M33.keys() if X in HST.keys()],            [M33[X]['DEC'] for X in M33.keys() if X in HST.keys()],            'o', color='k', mfc='blue', markeredgewidth=0.6, markersize=4, label='HST sample')


		plt.xlabel('RA', fontsize=13)
		plt.ylabel('DEC', fontsize=13)
		# plt.gca().invert_xaxis()
		plt.title('%s'%fil, fontsize=13)
		# plt.legend(fontsize=10)
		plt.show()

### Creates a file with all Cepheids and the 3 fourier modes to fit:
def create_file_fmodes():

	fparams_file = root_dat + 'fourier_modes_to_fit.dat'
	first_line = 'cep  mode_H 	mode_V 	mode_I	period      sample 	 \n'

	cepheids = [X for X in HST.keys() if (M33[X]['flag_pm11']=='3') and (M33[X]['sigma_phase']<0.05)]

	with open(fparams_file,"w") as mf:
		mf.write(first_line)
		for X in cepheids:
			
			sample_name = '---'
			if X in gold_sample:
				sample_name = 'go'
			elif X in silver_g_sample:
				sample_name = 'sg'
			elif X in silver_i_sample:
				sample_name = 'si'
			elif X in bronze_sample:
				sample_name = 'br'
			
			mf.write('%s  7   7   7  %.4f  	%s   \n'%(X, M33[X]['period'], sample_name))

###
def create_new_file_fmodes():
	fparams_file2 = root_dat + 'fourier_modes_to_fit2.dat'
	first_line = 'cep  mode_H 	mode_V 	mode_I	period      sample 	 \n'

	with open(fparams_file2,"w") as mf:
		mf.write(first_line)

		for i in range(len(fmodes_to_fit)):
			mf.write('%s  %i   %i   %i  %.4f  	%s   \n'%(fmodes_to_fit['cep'][i], fmodes_to_fit['mode_H'][i], fmodes_to_fit['mode_V'][i], fmodes_to_fit['mode_I'][i], fmodes_to_fit['period'][i], fmodes_to_fit['sample'][i]))


		for X in bronze_sample:
			if X not in list(fmodes_to_fit['cep']):
				mf.write('%s  7   7   7  %.4f  	br   \n'%(X, M33[X]['period']))

###
def replace_line(file_name, line_num, text):
    lines = open(file_name, 'r').readlines()
    lines[line_num] = text
    out = open(file_name, 'w')
    out.writelines(lines)
    out.close()

###
def which_sample(star):
	try:
		sample_name = fmodes_to_fit['sample'][list(fmodes_to_fit['cep']).index(star)]
	except:
		sample_name = 'br'

	if star in gold_sample_extended:
		sample_name = 'br'

	return(sample_name)

###
def mag2flux(mag_value, band):
	return(F_vega[band]*10**(-mag_value/2.5))

###
def flux2mag(flux_value, band):
	return(-2.5*np.log10(flux_value/F_vega[band]))





















###