from astropy.coordinates import ICRS, Distance, Angle
from astropy import units as u
import numpy as np
from astropy.io import ascii
import aplpy
from astroquery.skyview import SkyView
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from scipy.stats import chisquare
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
import math
import light_curve_fitting_M33 as LCfit

verboseTime=time.time()



###########################################################################################################################################################
root_grd = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/m33cep_ground/'
root_dat = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/'
root_LCs = '/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Figures_M33/LCs_cleaned/'

xfine = np.linspace(0.4, 2.2, 100) 
cfine = np.linspace(0.5, 2.6, 100) 
ofine = np.linspace(-0.5, 0., 100) 

LMC_DM = ufloat(18.477, 0.026)

CRNL = ufloat(0.0154, 0.005)
mygreen = (0.1, 0.75, 0.1)

d_M33 = 850
RA_M33, DEC_M33 = 23.46, 30.66
PA = 22
theta = PA+90
inclin = 53

outl1 = '01340120+3048131'
outl2 = '01334390+3032452'

gold_sample_extended = [] #['01333241+3031437', '01334104+3043399', '01334167+3043115', '01335067+3034459', '01335232+3046026', '01340058+3036306', '01341343+3043340', '01342241+3044080']

gradient = {'Magrini+09': [8.44, 0.06, -0.031, 0.013], 'Bresolin+11': [8.50, 0.02, -0.045, 0.006], 'Toribio+RL': [8.76, 0.07, -0.048, 0.019], 'Toribio+CEL': [8.52, 0.03, -0.053, 0.010], 'Lin+17': [8.46, 0.02, -0.025, 0.004], 'Rogers+22': [8.59, 0.02, -0.037, 0.007]}

### The fit did not converge, or output parameters (amplitude) too far from expected value form the ground...

### Load ground data (BVI, gri):
tab_grd = pd.read_csv(root_grd+'pm11.out', header=0, sep="\s+")
### Load HST data from Meredith:
tab_mered = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/meredith_cepheids_unstacked.dat', header=0, sep="\s+")
### Read second file Macri (with period uncertainties):
pm11b = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/pm11b.out', header=0, sep="\s+")
### Read Fourier modes to fit for each star (all HST ceps with "3" flag and sig(phase)<0.05):
fourier_modes_to_fit = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/fourier_modes_to_fit.dat', header=0, sep="\s+")
### Read output parameters:
res = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/output_params.dat', header=0, sep="\s+")

### Read phased data from Meredith:
f = open('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/meredith_cepheids_phased.dpy', 'rb')   
HST = pickle.load(f, encoding='latin1')
f.close()		

### Read M33 data from Lucas:
f = open('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/PM11_M33.dpy', 'rb')   
M33 = pickle.load(f, encoding='latin1')
f.close()	

abby = pd.read_csv('/Users/louise/Desktop/SH0ES/2022-09-19_M33_clusters/Data_M33/Cepheids_Abby.csv', header=0, sep=",")

### Stars with only one epoch per filter (or several points but too close, which makes to LC fit unsuccessful and unprecise):         
one_pt_per_filter = ['01333107+3031435', '01341586+3046125', '01333433+3034270', '01335346+3035354', '01335644+3047141', '01340071+3049589', '01340084+3049551', '01340474+3049181', '01340613+3037339', '01340955+3045357', '01341694+3046399', '01341754+3038196', '01341955+3049375', '01335911+3049049', '01333021+3041325', '01340498+3049285', '01332304+3032164', '01332556+3034273', '01332746+3037078', '01332875+3031316', '01332958+3031087', '01334139+3047364', '01334843+3050192', '01334880+3030448', '01341190+3029475', '01341955+3049375', '01342347+3042043', '01342385+3048589', '01343182+3043050', '01343169+3043002']
bad_LC_fit = ['01335279+3034164', '01335370+3031520', '01335613+3038039', '01335571+3044159', '01332847+3032351', '01332877+3037534', '01333884+3034230', '01334104+3043399', '01334159+3036095', '01334263+3033298', '01334306+3045014', '01334322+3032434', '01334362+3047100', '01334398+3045570', '01334462+3044151', '01334645+3047405', '01335099+3031564', '01335305+3045414', '01335558+3044484', '01335668+3048378', '01335747+3049072', '01335894+3045307', '01342861+3048207', '01341720+3047514', '01341215+3043528', '01340864+3037547', ]
divergent_amp = [res['cep'][i] for i in range(len(res)) if (abs(M33[res['cep'][i]]['gamp'][0] - res['Av'][i])>0.5) and (M33[res['cep'][i]]['gamp'][0]<9.)]

outliers = []
for X in one_pt_per_filter:
	if X not in outliers:
		outliers.append(X)
for X in bad_LC_fit:
	if X not in outliers:
		outliers.append(X)
for X in divergent_amp:
	if X not in outliers:
		outliers.append(X) 

clusters_ceps = ['01342988+3047541', '01341217+3036362', '01340959+3036215', '01340910+3036296', '01335809+3045568', '01335311+3048343', '01334821+3038001', '01334331+3043559', '01333438+3035307', '01333348+3033210', '01332768+3034238']

###################################################################################################################################################
### MAIN FUNCTIONS:
###
def PL_mean_mags(SAMPLE, showPL):

	### THE "GOLD+SILVER" SAMPLE IN THE CODE IS "GOLD" IN THE PAPER;
	### THE "GOLD+SILVER+BRONZE" IN THE CODE IS "GOLD+SILVER" IN THE PAPER.

	### Gold sample:
	if SAMPLE == 'G':
		indices = [i for i in range(len(res)) if (res['sample'][i] == 'go') and (res['cep'][i] not in outliers)]
	### Extended gold sample + silver sample
	elif SAMPLE == 'GS':
		indices = [i for i in range(len(res)) if (res['sample'][i] in ['go', 'si', 'sg']) and (res['cep'][i] not in outliers)]
	elif SAMPLE == 'GSB':
		indices = [i for i in range(len(res)) if (res['sample'][i] in ['go', 'si', 'sg', 'br']) and (res['cep'][i] not in outliers)]

	cep           = [res['cep'][i]                                                         for i in indices]
	chi2          = [res['chi2'][i]                                                        for i in indices]
	logP          = [np.log10(res['period'][i])                                            for i in indices]
	color_phatter = [ufloat(res['V'][i], res['eV'][i]) - ufloat(res['I'][i], res['eI'][i]) for i in indices]
	color_sh0es   = [0.065 + 0.658*col for col in color_phatter]
	Hmag          = [res['H'][i]                                                           for i in indices]
	eHmag         = [res['eH'][i]                                                          for i in indices]
	samp_name     = [res['sample'][i]                                                      for i in indices]
	delta_mag     = [M33[res['cep'][i]]['delta_mag']                                       for i in indices]
	print(' Sample: %s     N = %i'%(SAMPLE, len(indices)))

	### WH from HST/WFC3 magnitudes, random-phase corrected:	
	wH_sh0es   = [ufloat(Hmag[i], eHmag[i]) + delta_mag[i] - 0.386*color_sh0es[i] for i in range(len(indices))]
	mH, emH = [y.nominal_value for y in wH_sh0es], [y.std_dev for y in wH_sh0es]
	### Adding intrinsic scatter of the PL to WH errors (see Li+ Table 3, footnote):
	emH = [np.sqrt(em**2 + 0.069**2) for em in emH]

	### Free slope:
	popt, pcov = curve_fit(lambda X, A, B: A*X+B, logP, mH, sigma=emH)
	perr = np.sqrt(np.diag(pcov))
	slope, zp = popt[0], popt[1]
	e_slope, e_zp = perr[0], perr[1]
	sigma = np.sqrt(sum( (np.array(mH)-np.array([slope*XX + zp for XX in logP]))**2 )/len(indices))
	fit = np.array([slope*XX + zp for XX in logP])
	print('PL (free slope):  %.3f ± %.3f \n'%(zp, e_zp))

	chi2 = sum([(fit[i] - mH[i])**2/(emH[i]**2) for i in range(len(indices))])  ### (np.array(mH)-np.array([slope*XX + zp for XX in logP]))**2 / np.array(mH)
	chi2r = chi2/(len(indices)-2)
	print(' ( chi2_dof = %.3f )'%chi2r)

	### Fixed slope (R19):
	popt_fix, pcov_fix = curve_fit(lambda X, B: -3.26*X+B, logP, mH, sigma=emH)
	perr_fix = np.sqrt(np.diag(pcov_fix))
	slope_fix, zp_fix = -3.26, popt_fix[0]
	e_slope_fix, e_zp_fix = 0, perr_fix[0]
	print('PL (fixed slope): %.3f ± %.3f \n'%(zp_fix, e_zp_fix))

	if showPL == True:

		plt.figure(figsize=(10,5))
		plt.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.10, hspace=0.1, wspace=0.3)

		# plt.plot(xfine, slope_gold*(xfine)+zp_gold, '-', linewidth=1.2, color='darkblue', label='$m_H^W$ (gold) $= %.3f_{\pm %.3f} \, \log P + %.3f_{\pm %.3f}$ ($\sigma = %.4f$)'%(slope_gold, e_slope_gold, zp_gold, e_zp_gold, sigma))
		plt.plot(xfine, slope*(xfine)+zp, '-', linewidth=1.2, color='darkblue', label='$m_H^W$ $= %.3f_{\pm %.3f} \, \log P + %.3f_{\pm %.3f}$ ($\sigma = %.3f$)'%(slope, e_slope, zp, e_zp, sigma))
		plt.plot(xfine, -3.26*(xfine)+22.038, '--', linewidth=1.2, color='crimson', label='Slope fixed to -3.26 (Riess+19)')
		
		if 'B' in SAMPLE:
			plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]=='br'], [mH[i] for i in range(len(cep)) if samp_name[i]=='br'], 
				yerr=[emH[i] for i in range(len(cep)) if samp_name[i]=='br'], fmt='o', color='lightgray', markerfacecolor='lightgray', 
				alpha=1.,  markeredgewidth=0.5, markersize=5, capsize=0, ecolor='lightgray', elinewidth=0.7, label='Silver sample')

		plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i] != 'br'], [mH[i] for i in range(len(cep)) if samp_name[i] != 'br'], 
			yerr=[emH[i] for i in range(len(cep)) if samp_name[i] != 'br'], fmt='o', color='k', markerfacecolor='darkblue', 
			markeredgewidth=0.5, markersize=5, capsize=0, ecolor='darkblue', elinewidth=0.7, label='Silver sample')

		# plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]=='go'], [mH[i] for i in range(len(cep)) if samp_name[i]=='go'], 
		# 	yerr=[emH[i] for i in range(len(cep)) if samp_name[i]=='go'], fmt='o', color='k', markerfacecolor='darkblue', 
		# 	markeredgewidth=0.5, markersize=7, capsize=0, ecolor='darkblue', elinewidth=0.7, label='Gold sample')

		# if 'B' in SAMPLE:
		# 	plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]=='br'], [mH[i] for i in range(len(cep)) if samp_name[i]=='br'], 
		# 		yerr=[emH[i] for i in range(len(cep)) if samp_name[i]=='br'], fmt='o', color='k', markerfacecolor='red', 
		# 		alpha=1.,  markeredgewidth=0.5, markersize=5, capsize=0, ecolor='red', elinewidth=0.7)#, label='Silver sample')

		# plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]!='br'], [mH[i] for i in range(len(cep)) if samp_name[i]!='br'], 
		# 	yerr=[emH[i] for i in range(len(cep)) if samp_name[i]!='br'], fmt='o', color='k', markerfacecolor='red', 
		# 	markeredgewidth=0.5, markersize=5, capsize=0, ecolor='red', elinewidth=0.7, label='HST photometry ($\sigma = 0.11$ mag)')


		plt.xlabel('$\log P$ (days)', fontsize=11)
		plt.ylabel('$m_H^W$ (mag)', fontsize=11)
		plt.gca().invert_yaxis()
		plt.xlim(0.45, 1.95)
		plt.ylim(20.6, 15.3)
		plt.legend(fontsize=10, loc='upper left', fancybox=True, shadow=True)
		plt.show()

	return(-3.26, 0., zp_fix, e_zp_fix)

###
def distance_to_M33(SAMPLE):

	### PL relation in M33 (this work):
	slope_M33, e_slope_M33, zp_M33, e_zp_M33 = PL_mean_mags(SAMPLE, showPL=False)
	# zp_M33, e_zp_M33 = 22.059, 0.008

	### PL relation in the LMC (Riess+2019):
	zp_LMC, e_zp_LMC = 15.898, 0.009
	### Metallicity effect (Riess+2022):
	gamma = ufloat(-0.217, 0.046)
	### Metallicity of LMC and (mean) M33 from Bresolin+2011 gradient:
	OH_LMC = ufloat(-0.32, 0.01)
	# OH_LMC = ufloat(-0.32, 0.0)
	OH_M33_sample_uf = [ufloat(-0.045, 0.006)*M33[X]['d_galac'] + ufloat(8.50, 0.02) - 8.69 for X in HST.keys()]
	OH_M33 = np.mean([o.nominal_value for o in OH_M33_sample_uf])
	### Derive M33 distance modulus:
	distance_modulus = LMC_DM + ufloat(zp_M33, e_zp_M33) - (ufloat(zp_LMC, e_zp_LMC) + CRNL) - gamma*(ufloat(OH_M33, 0.03) - OH_LMC)
	DM_M33, e_DM_M33 = distance_modulus.nominal_value, distance_modulus.std_dev

	distance_kpc = 10**((distance_modulus-10)/5)
	dist_kpc, e_dist_kpc = distance_kpc.nominal_value, distance_kpc.std_dev

	print('\n Metallicity correction (gamma = %.3f mag/dex): \n -> delta(m) = %.3f mag'%(gamma.nominal_value, (- gamma*(OH_M33 - OH_LMC)).nominal_value ))

	print('\n distance (M33) = %.4f ± %.4f mag'%(DM_M33, e_DM_M33))
	print('                = %.0f ± %.0f kpc (%.3f percent)'%(dist_kpc, e_dist_kpc, e_dist_kpc/dist_kpc*100))

	#########################################################################################################################################################
	ref   = ['Freedman', 'Freedman', 'Macri', 'McConnachie', 'Bonanos', 'Sarajedini', 'Rizzi', 'Scowcroft', 'U',   'Pellerin', 'Conn', 'Gieren', 'Bhardwaj', 'Yuan', 'Zgirski', 'Lee',  'Lee',  'Lee',  'Ou',  'Breuval' ]
	year  = [ 1991,       2001,       2001,    2004,          2006,      2006,         2007,    2009,        2009,  2011,       2012,   2013,     2016,       2018,   2021,      2022,   2022,   2022,   2023,  2023     ]
	dist  = [ 24.64,      24.56,      24.65,   24.54,         24.92,     24.67,        24.71,   24.53,       24.84, 24.76,      24.57,  24.62,    24.62,      24.80,  24.57,     24.71,  24.72,  24.67,  24.67, DM_M33   ]
	edist = [ 0.09,       0.10,       0.12,    0.06,          0.12,      0.08,         0.04,    0.11,        0.10,  0.05,       0.05,   0.07,     0.06,       0.06,   0.06,      0.04,   0.07,   0.05,   0.06,  e_DM_M33 ]
	mtd   = ['C',        'C',        'C',     'T',           'D',       'R',          'T',     'C',         'T',   'C',        'T',    'C',      'C',        'M',    'J',       'C',    'T',    'J',    'M',   'C'       ]
	y_ax  = [ 0,          1,          2,       3,             4,         5,            6.,      7.,          8,     9,          10,     11,       12,         13,     14,        14.82,  15.18,  15,     16,    17       ]
	LMC   = [ 18.5,       18.5,       18.5,    18.477,        18.477,    18.5,         18.5,    18.4,        18.5,  18.5,       18.5,   18.5,     18.47,      18.493, 18.477,    18.477, 18.477, 18.477, 18.49, 18.477   ]
	rescaled_dist = [18.477-LMC[i]+dist[i] for i in range(len(year)) ]

	clr = {'C': 'royalblue', 'T': 'firebrick', 'J': 'gold', 'M': 'coral', 'D': 'lightskyblue', 'R': 'yellowgreen'}

	# print('\n Ref 		dist 	edist 	indicator 	d_LMC')
	# for i in range(len(ref)):
	# 	print('%s_%i  		%.3f 	%.3f 	%s 	%.3f'%(ref[i], year[i], dist[i], edist[i], mtd[i], LMC[i]  ))

	wmean_mu_C = np.sum([rescaled_dist[i]/edist[i] for i in range(len(dist)) if mtd[i]=='C']) / np.sum([1/edist[i] for i in range(len(dist)) if mtd[i]=='C'])
	wmean_mu_T = np.sum([rescaled_dist[i]/edist[i] for i in range(len(dist)) if mtd[i]=='T']) / np.sum([1/edist[i] for i in range(len(dist)) if mtd[i]=='T'])
	wmean_mu_J = np.sum([rescaled_dist[i]/edist[i] for i in range(len(dist)) if mtd[i]=='J']) / np.sum([1/edist[i] for i in range(len(dist)) if mtd[i]=='J'])
	wmean_mu_M = np.sum([rescaled_dist[i]/edist[i] for i in range(len(dist)) if mtd[i]=='M']) / np.sum([1/edist[i] for i in range(len(dist)) if mtd[i]=='M'])

	chi2_wght_C = np.sum([(rescaled_dist[i]-wmean_mu_C)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='C']) / (len([x for x in mtd if x=='C']) - 1)
	chi2_wght_T = np.sum([(rescaled_dist[i]-wmean_mu_T)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='T']) / (len([x for x in mtd if x=='T']) - 1)
	chi2_wght_J = np.sum([(rescaled_dist[i]-wmean_mu_J)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='J']) / (len([x for x in mtd if x=='J']) - 1)
	chi2_wght_M = np.sum([(rescaled_dist[i]-wmean_mu_M)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='M']) / (len([x for x in mtd if x=='M']) - 1)

	chi2_best_C = np.sum([(rescaled_dist[i]-DM_M33)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='C' and ref[i]!='Breuval']) / (len([x for x in mtd if x=='C']) - 1)
	chi2_best_T = np.sum([(rescaled_dist[i]-DM_M33)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='T']) / (len([x for x in mtd if x=='T']))
	chi2_best_J = np.sum([(rescaled_dist[i]-DM_M33)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='J']) / (len([x for x in mtd if x=='J']))
	chi2_best_M = np.sum([(rescaled_dist[i]-DM_M33)**2/(edist[i])**2 for i in range(len(dist)) if mtd[i]=='M']) / (len([x for x in mtd if x=='M']))

	print('\nweighted mean Cepheids: %.3f mag   chi2_wght = %.2f    chi2_best = %.2f '%(wmean_mu_C, chi2_wght_C, chi2_best_C))
	print('weighted mean TRGB:     %.3f mag   chi2_wght = %.2f    chi2_best = %.2f  '%(wmean_mu_T, chi2_wght_T, chi2_best_T))
	print('weighted mean JAGB:     %.3f mag   chi2_wght = %.2f    chi2_best = %.2f  '%(wmean_mu_J, chi2_wght_J, chi2_best_J))
	print('weighted mean Miras:    %.3f mag   chi2_wght = %.2f    chi2_best = %.2f  '%(wmean_mu_M, chi2_wght_M, chi2_best_M))

	plt.figure(figsize=(6,7))
	plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.07, hspace=0.1, wspace=0.3)
	### Fill area 1-sigma from present work:
	plt.fill_between([DM_M33-e_DM_M33, DM_M33+e_DM_M33], [-1., -1.], [24, 24], facecolor='darkblue', alpha=0.12)
	### Plots for label:
	plt.errorbar(18.477-LMC[mtd.index('C')]+dist[mtd.index('C')], y_ax[mtd.index('C')], xerr=edist[mtd.index('C')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('C')]], label='Cepheids', alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('C')]], elinewidth=1.4)
	plt.errorbar(18.477-LMC[mtd.index('D')]+dist[mtd.index('D')], y_ax[mtd.index('D')], xerr=edist[mtd.index('D')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('D')]], label='DEBs',     alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('D')]], elinewidth=1.4)
	plt.errorbar(18.477-LMC[mtd.index('R')]+dist[mtd.index('R')], y_ax[mtd.index('R')], xerr=edist[mtd.index('R')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('R')]], label='RR Lyrae', alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('R')]], elinewidth=1.4)
	plt.errorbar(18.477-LMC[mtd.index('T')]+dist[mtd.index('T')], y_ax[mtd.index('T')], xerr=edist[mtd.index('T')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('T')]], label='TRGB',     alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('T')]], elinewidth=1.4)
	plt.errorbar(18.477-LMC[mtd.index('M')]+dist[mtd.index('M')], y_ax[mtd.index('M')], xerr=edist[mtd.index('M')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('M')]], label='Miras',    alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('M')]], elinewidth=1.4)
	plt.errorbar(18.477-LMC[mtd.index('J')]+dist[mtd.index('J')], y_ax[mtd.index('J')], xerr=edist[mtd.index('J')], fmt='o', color='k', markerfacecolor=clr[mtd[mtd.index('J')]], label='JAGB',     alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[mtd.index('J')]], elinewidth=1.4)

	for i in range(len(year)-1):
		plt.errorbar(rescaled_dist[i], y_ax[i], xerr=edist[i], fmt='o', color='k', markerfacecolor=clr[mtd[i]], alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr[mtd[i]], elinewidth=1.4)
		if y_ax[i]%1 == 0:
			plt.annotate('%s+%i'%(ref[i], year[i]), xy=(dist[i], y_ax[i]), xytext=(24.15, y_ax[i]+0.2), fontsize=10, color='k') 

	### Plot present work:
	plt.errorbar(DM_M33, y_ax[-2]+1, xerr=e_DM_M33, fmt='o', color='k', markerfacecolor=clr['C'], alpha=1.0, markeredgewidth=0.8, markersize=6, capsize=0, ecolor=clr['C'], elinewidth=1.4)
	plt.annotate('Present work', xy=(DM_M33, y_ax[-2]+1), xytext=(24.15, y_ax[-2]+1+0.2), fontsize=10, color='k', weight='bold') 
	plt.annotate(r'($\mu_{\rm M33} = %.3f \pm %.3f$ mag)'%(DM_M33, e_DM_M33), xy=(DM_M33, y_ax[-2]+1), xytext=(24.15, y_ax[-2]+1+0.7), fontsize=10, color='k') 

	plt.xlabel('M33 distance modulus (mag)', fontsize=11)
	plt.xlim(24.12, 25.07)
	plt.ylim(-0.7, y_ax[-2]+2+0.2)
	plt.legend(loc='lower right', fontsize=11, fancybox=True, shadow=True)
	plt.gca().set_yticks([])
	plt.gca().invert_yaxis()
	plt.show()


###################################################################################################################################################
###
def plot_output_Av():

	plt.figure(figsize=(8,5))
	plt.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.12, hspace=0.1, wspace=0.3)

	plt.fill_between([0., 2.], [-0.2, -0.2], [0.2, 0.2], facecolor='green', alpha=0.2)


	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'go'], [output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0]      for i in range(len(output_params)) if output_params['sample'][i]=='go'], 'o', color='k', markerfacecolor='gold', markeredgewidth=0.8, markersize=6, label='gold sample')
	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'sg'], [output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0]      for i in range(len(output_params)) if output_params['sample'][i]=='sg'], 'o', color='k', markerfacecolor='deepskyblue', markeredgewidth=0.8, markersize=6, label='silver "g" sample')
	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'si'], [output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['iamp'][0]/0.58 for i in range(len(output_params)) if output_params['sample'][i]=='si'], 'o', color='k', markerfacecolor='green', markeredgewidth=0.8, markersize=6, label='silver "i" sample')

	# for i in range(len(output_params)):
	# 	if (output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0] > 0.5) or (output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0] < -0.2):
	# 		plt.annotate('%s (%i)'%(output_params['cep'][i], i), xy=(M33[output_params['cep'][i]]['logP'], output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0]), xytext=(M33[output_params['cep'][i]]['logP']+0.01, output_params['Av'][i]-0.5*M33[output_params['cep'][i]]['gamp'][0]), fontsize=6, color='k') 

	plt.xlabel('$\log P$ (days)', fontsize=11)
	plt.ylabel(r'$A_V^{\rm final} - A_V^{\rm expected}$ (mag)', fontsize=11)
	plt.xlim(0.3, 2.)
	plt.ylim(-0.3, 0.3)
	plt.legend(loc='best')
	plt.show()

###
def plot_output_dphase():

	plt.figure(figsize=(8,5))
	plt.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.12, hspace=0.1, wspace=0.3)

	plt.fill_between([0., 2.], [-0.05, -0.05], [0.05, 0.05], facecolor='green', alpha=0.2)

	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'go'], [output_params['phase'][i] for i in range(len(output_params)) if output_params['sample'][i]=='go'], 'o', color='k', markerfacecolor='gold', markeredgewidth=0.8, markersize=6, label='gold sample')
	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'si'], [output_params['phase'][i] for i in range(len(output_params)) if output_params['sample'][i]=='si'], 'o', color='k', markerfacecolor='deepskyblue', markeredgewidth=0.8, markersize=6, label='silver "g" sample')
	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'sg'], [output_params['phase'][i] for i in range(len(output_params)) if output_params['sample'][i]=='sg'], 'o', color='k', markerfacecolor='green', markeredgewidth=0.8, markersize=6, label='silver "i" sample')
	plt.plot([M33[output_params['cep'][i]]['logP'] for i in range(len(output_params)) if output_params['sample'][i] == 'br'], [output_params['phase'][i] for i in range(len(output_params)) if output_params['sample'][i]=='br'], 'o', color='k', markerfacecolor='darkred', markeredgewidth=0.8, markersize=6, label='bronze sample')

	plt.xlabel('$\log P$ (days)', fontsize=11)
	plt.ylabel(r'$\Delta \phi$', fontsize=11)
	plt.xlim(0.3, 2.)
	plt.ylim(-0.1, 0.1)
	plt.legend(loc='best')
	plt.show()

###
def gamma_M33(gradient_ref):

	indices = [i for i in range(len(res)) if (res['cep'][i] not in outliers)]

	cep           = [res['cep'][i]                                                                         for i in indices]
	logP          = [np.log10(res['period'][i])                                                            for i in indices]
	color_sh0es   = [0.065 + 0.658*(ufloat(res['V'][i], res['eV'][i]) - ufloat(res['I'][i], res['eI'][i])) for i in indices]
	Hmag          = [res['H'][i]                                                                           for i in indices]
	eHmag         = [res['eH'][i]                                                                          for i in indices]
	delta_mag     = [M33[res['cep'][i]]['delta_mag']                                                       for i in indices]

	### WH from HST/WFC3 magnitudes, random-phase corrected:	
	wH_sh0es   = [ufloat(Hmag[i], eHmag[i]) + delta_mag[i] - 0.386*color_sh0es[i] for i in range(len(indices))]
	mH  = [y.nominal_value for y in wH_sh0es]
	emH = [np.sqrt(y.std_dev**2 + 0.069**2) for y in wH_sh0es]			### Adding intrinsic scatter

	### Fitting free PL:
	popt, pcov = curve_fit(lambda X, A, B: A*X+B, logP, mH, sigma=emH)
	perr = np.sqrt(np.diag(pcov))
	slope, zp = popt[0], popt[1]
	e_slope, e_zp = perr[0], perr[1]

	sigma = np.sqrt(sum( (np.array(mH)-np.array([slope*XX + zp for XX in logP]))**2 )/len(indices))
	residuals   = [slope*logP[i]+zp - mH[i] for i in range(len(cep))]

	### Gradient:
	bg, ebg, ag, eag = gradient[gradient_ref]
	OH_uf = [ufloat(bg, ebg) + ufloat(ag, eag)*M33[X]['d_galac'] -8.69 for X in cep]
	OH, eOH = [x.nominal_value for x in OH_uf], [x.std_dev for x in OH_uf]

	### Fitting metallicity effect:
	popt1, pcov1 = curve_fit(lambda X, G, D: G*X+D, OH, residuals, sigma=[np.sqrt(eOH[i]**2 + emH[i]**2) for i in range(len(cep))])
	perr1 = np.sqrt(np.diag(pcov1))
	gamma, delta = popt1[0], popt1[1]
	e_gamma, e_delta = perr1[0], perr1[1]

	plt.figure(figsize=(7,4))
	plt.subplots_adjust(left=0.10, right=0.96, top=0.92, bottom=0.12, hspace=0.1, wspace=0.3)
	plt.plot([-0.8, 0.5], [0., 0.], '--', linewidth=1.2, color='grey', alpha=1.)

	plt.plot(ofine, gamma*(ofine)+delta, '-', linewidth=1.2, color='b', alpha=1., label='$\gamma = %.3f \pm %.3f$ mag/dex'%(-gamma, e_gamma))
	
	plt.errorbar(OH, residuals, xerr=eOH, yerr=emH, fmt='o', color='k', markerfacecolor='b', alpha=0.4, markeredgewidth=0.5, markersize=7, capsize=0, ecolor='b', elinewidth=1.)
		
	plt.xlabel('[O/H] (dex)', fontsize=11)
	plt.ylabel('PL residuals (mag)', fontsize=11)
	plt.xlim(np.mean(OH)-0.25, np.mean(OH)+0.25)
	plt.ylim(np.mean(residuals)-0.6, np.mean(residuals)+0.6)
	plt.gca().invert_xaxis()
	plt.legend(fontsize=11, loc='lower left', fancybox=True, shadow=True)
	plt.title('%s'%gradient_ref)
	plt.show()	

	print('\n Gradient from %s:  %.2f < [O/H] < %.2f dex  (delta = %.2f) \n  ---> gamma = %.3f ± %.3f mag/dex'%(gradient_ref, min(OH), max(OH), abs(min(OH)-max(OH)), -gamma, e_gamma))

### 
def ACS_to_WFC3():

	max_logg = 2
	min_mass = 3
	max_mass = 7
	min_T = 4800
	max_T = 6500

	print('\n Loading WFC3 isochrones...')
	WFC = ascii.read(root_dat+'/isochrones/isochrones_WFC3_01Av.dat')
	ind_WFC = [i for i in range(len(WFC)) if (WFC['logg'][i] < max_logg) and (WFC['Mass'][i] > min_mass) and (WFC['Mass'][i] < max_mass) and (10**(WFC['logTe'][i]) > min_T) and (10**(WFC['logTe'][i]) < max_T)]

	print(' Loading ACS isochrones...')
	ACS = ascii.read(root_dat+'/isochrones/isochrones_ACS_01Av.dat')
	ind_ACS = [i for i in range(len(ACS)) if (ACS['logg'][i] < max_logg) and (ACS['Mass'][i] > min_mass) and (ACS['Mass'][i] < max_mass) and (10**(ACS['logTe'][i]) > min_T) and (10**(ACS['logTe'][i]) < max_T)]

	
	### (1) Comparing ACS PHATTER color (F475W-F814W) and WFC3 SH0ES color (F555W-F814W):
	color_sh0es   = [WFC['F555Wmag'][i] - WFC['F814Wmag'][i] for i in ind_WFC]
	color_phatter = [ACS['F475Wmag'][i] - ACS['F814Wmag'][i] for i in ind_ACS]
	popt, pcov = curve_fit(lambda X, A, B: A*X+B, color_phatter, color_sh0es)
	slope, zp = popt[0], popt[1]
	sigma = np.sqrt(sum( (np.array(color_sh0es)-np.array([slope*XX + zp for XX in color_phatter]))**2 )/len(color_sh0es))
	plt.figure(figsize=(8,5))
	plt.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.10, hspace=0.1, wspace=0.3)
	# plt.plot(cfine, 0.666*cfine-0.111, '--', linewidth=0.9, color='m', alpha=1., label=r"$(F555W-F814W)_{\rm WFC3} = -0.111 + 0.666 \, (F475W-F814W)_{\rm ACS}$ (Adam)")
	plt.plot(cfine, slope*cfine+zp, '-', linewidth=0.9, color='b', alpha=1., label=r'$(F555W-F814W)_{\rm WFC3} = +%.3f + %.3f \, (F475W-F814W)_{\rm ACS} \, \, \, (\sigma = %.3f)$'%(zp, slope, sigma))
	plt.plot(color_phatter, color_sh0es, 'o', color='k', markerfacecolor='b', alpha=0.4, markeredgewidth=0.5, markersize=5)
	plt.xlabel('ACS (F475W-F814W)', fontsize=11)
	plt.ylabel('WFC3 (F555W-F814W)', fontsize=11)
	plt.xlim(0.6, 2.0)
	plt.ylim(0.4, 1.4)
	plt.legend(fontsize=10, loc='upper left', fancybox=True, shadow=True)
	plt.title(' (log(g) < %.0f)     (%.0f < M < %.0f M$_{sun}$)       (%.0f < T < %.0f) '%(max_logg, min_mass, max_mass, min_T, max_T))
	plt.show()
	
### Map with colormap markers:
def map_M33_HST():

	color_map = plt.cm.get_cmap('viridis')
	# color_map = color_map.reversed()

	print(' Downloading M33 image ...')
	paths = SkyView.get_images(position=SkyCoord(23.46*u.deg, 30.66*u.deg, frame='icrs'), survey=['DSS1 Red'], radius=0.9*u.deg, pixels=1200)
	fig = aplpy.FITSFigure(paths[0], auto_refresh=False, figsize=(10, 8)) 
	fig.set_xaxis_coord_type('scalar')
	fig.set_yaxis_coord_type('scalar')
	fig.show_colorscale(pmin=40,pmax=99, stretch='log', cmap='gist_yarg')

	fig.show_markers([M33[X]['RA'] for X in M33.keys()], [M33[X]['DEC'] for X in M33.keys()], marker='s', facecolor='none', edgecolor='crimson', linewidth=1.0, s=30)
	for X in HST.keys():
		fig.show_markers(M33[X]['RA'], M33[X]['DEC'], marker='D', facecolor='none', edgecolor='deepskyblue', linewidth=1.0, s=12)
		if X in list(res['cep']):
			fig.show_markers(M33[X]['RA'], M33[X]['DEC'], marker='D', facecolor='deepskyblue', edgecolor='deepskyblue', linewidth=1.0, s=12)
		if (X in list(res['cep'])) and (res['sample'][list(res['cep']).index(X)] != 'br'):
			fig.show_markers(M33[X]['RA'], M33[X]['DEC'], marker='D', facecolor='blue', edgecolor='blue', linewidth=1.0, s=12)
	### Show Abby's Cepheids:
	# for i in range(len(abby)):
	# 	fig.show_markers(abby['ra'][i], abby['dec'][i], marker='X', facecolor='lime', edgecolor='k', linewidth=0.6, s=26)

	fig.show_markers(23.4418690, 30.7441144, marker='X', facecolor='gold', edgecolor='k', linewidth=0.7, s=26)

	plt.title('M33')
	plt.show()

###
def show_filters_transmission_curves():

	g_CFHT = ascii.read(root_dat+'filters_transmission_curves/CFHT_MegaCam_g.dat')
	r_CFHT = ascii.read(root_dat+'filters_transmission_curves/CFHT_MegaCam_r.dat')
	i_CFHT = ascii.read(root_dat+'filters_transmission_curves/CFHT_MegaCam_i.dat')

	F160W = ascii.read(root_dat+'filters_transmission_curves/HST_F160W.dat')
	F475W = ascii.read(root_dat+'filters_transmission_curves/HST_F475W.dat')
	F814W = ascii.read(root_dat+'filters_transmission_curves/HST_F814W.dat')
	F555W = ascii.read(root_dat+'filters_transmission_curves/HST_F555W.dat')

	J_2mass = ascii.read(root_dat+'filters_transmission_curves/2MASS_2MASS_J.dat')
	H_2mass = ascii.read(root_dat+'filters_transmission_curves/2MASS_2MASS_H.dat')
	K_2mass = ascii.read(root_dat+'filters_transmission_curves/2MASS_2MASS_Ks.dat')

	plt.figure(figsize=(7.5, 7)) 
	gs=gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])
	gs.update(wspace=0.25, hspace=0.65, left=0.01, bottom=0.09, right=0.99, top=0.94)
	ax1, ax2, ax3 = plt.subplot(gs[0]), plt.subplot(gs[1]), plt.subplot(gs[2])


	ax1.fill_between([x/10000 for x in F160W['col1']], [0. for y in F160W['col2']], [y for y in F160W['col2']], facecolor='red', alpha=0.2)
	ax1.fill_between([x/10000 for x in F475W['col1']], [0. for y in F475W['col2']], [y for y in F475W['col2']], facecolor='indigo', alpha=0.2)
	ax1.fill_between([x/10000 for x in F814W['col1']], [0. for y in F814W['col2']], [y for y in F814W['col2']], facecolor=mygreen, alpha=0.2)
	ax1.fill_between([x/10000 for x in F555W['WAVELENGTH']], [0. for y in F555W['THROUGHPUT']], [y for y in F555W['THROUGHPUT']], facecolor='blue', alpha=0.2)
	ax1.plot(F160W['col1']/10000, F160W['col2'], '-', color='red', linewidth=2.)
	ax1.plot(F475W['col1']/10000, F475W['col2'], '-', color='indigo', linewidth=2.)
	ax1.plot(F814W['col1']/10000, F814W['col2'], '-', color=mygreen, linewidth=2.)
	ax1.plot(F555W['WAVELENGTH']/10000, F555W['THROUGHPUT'], '-', color='blue', linewidth=2.)
	ax1.annotate('F160W', xy=((np.mean(F160W['col1'])-900)/10000, 0.6), xytext=((np.mean(F160W['col1'])-900)/10000, 0.6), fontsize=10, color='red', weight='bold') 
	ax1.annotate('F475W', xy=((np.mean(F475W['col1'])-900)/10000, 0.6), xytext=((np.mean(F475W['col1'])-700)/10000, 0.42), fontsize=10, color='indigo', weight='bold') 
	ax1.annotate('F814W', xy=((np.mean(F814W['col1'])-900)/10000, 0.6), xytext=((np.mean(F814W['col1'])-900)/10000, 0.5), fontsize=10, color=mygreen, weight='bold') 
	ax1.annotate('F555W', xy=((np.mean(F555W['WAVELENGTH'])-900)/10000, 0.6), xytext=((np.mean(F555W['WAVELENGTH'])-100)/10000, 0.35), fontsize=10, color='blue', weight='bold') 
	ax1.set_ylim(0., 0.7)
	ax1.set_xlim(3700/10000, 18500./10000)
	ax1.set_title('HST', weight='bold')
	ax1.tick_params(left = False)


	ax2.fill_between([x/10000 for x in g_CFHT['col1']], [0. for y in g_CFHT['col2']], [y for y in g_CFHT['col2']], facecolor='indigo', alpha=0.2)
	ax2.fill_between([x/10000 for x in r_CFHT['col1']], [0. for y in r_CFHT['col2']], [y for y in r_CFHT['col2']], facecolor='dodgerblue', alpha=0.2)
	ax2.fill_between([x/10000 for x in i_CFHT['col1']], [0. for y in i_CFHT['col2']], [y for y in i_CFHT['col2']], facecolor=mygreen, alpha=0.2)
	ax2.plot(g_CFHT['col1']/10000, g_CFHT['col2'], '-', color='indigo', linewidth=2.)
	ax2.plot(r_CFHT['col1']/10000, r_CFHT['col2'], '-', color='dodgerblue', linewidth=2.)
	ax2.plot(i_CFHT['col1']/10000, i_CFHT['col2'], '-', color=mygreen, linewidth=2.)
	ax2.annotate('g', xy=((np.mean(g_CFHT['col1'])-200)/10000, 0.58), xytext=((np.mean(g_CFHT['col1'])-200)/10000, 0.58), fontsize=12, color='indigo', weight='bold') 
	ax2.annotate('r', xy=((np.mean(r_CFHT['col1'])-200)/10000, 0.58), xytext=((np.mean(r_CFHT['col1'])-100)/10000, 0.56), fontsize=12, color='dodgerblue', weight='bold') 
	ax2.annotate('i', xy=((np.mean(i_CFHT['col1'])-200)/10000, 0.58), xytext=((np.mean(i_CFHT['col1'])-200)/10000, 0.52), fontsize=12, color=mygreen, weight='bold') 
	ax2.set_ylim(0., 0.67)
	ax2.set_xlim(3700/10000, 18500./10000)
	ax2.set_title('CFHT/MegaCam', weight='bold')
	ax2.tick_params(left = False)


	ax3.fill_between([x/10000 for x in J_2mass['col1']], [0. for y in J_2mass['col2']], [y for y in J_2mass['col2']], facecolor='orange', alpha=0.2)
	ax3.fill_between([x/10000 for x in H_2mass['col1']], [0. for y in H_2mass['col2']], [y for y in H_2mass['col2']], facecolor='darkred', alpha=0.2)
	# ax3.fill_between([x for x in K_2mass['col1']], [0. for y in K_2mass['col2']], [y for y in K_2mass['col2']], facecolor='darkred', alpha=0.2)
	ax3.plot(J_2mass['col1']/10000, J_2mass['col2'], '-', color='orange', linewidth=2.)
	ax3.plot(H_2mass['col1']/10000, H_2mass['col2'], '-', color='darkred', linewidth=2.)
	# ax3.plot(K_2mass['col1'], K_2mass['col2'], '-', color='darkred', linewidth=2.)
	ax3.annotate('J', xy=((np.mean(J_2mass['col1'])-200)/10000, 1.1), xytext=((np.mean(J_2mass['col1'])-200)/10000, 1.1), fontsize=11, color='orange', weight='bold') 
	ax3.annotate('H', xy=((np.mean(H_2mass['col1'])-200)/10000, 1.1), xytext=((np.mean(H_2mass['col1'])-200)/10000, 1.1), fontsize=11, color='darkred', weight='bold') 
	# ax3.annotate('K', xy=(np.mean(K_2mass['col1'])-200, 1.1), xytext=(np.mean(K_2mass['col1'])-200, 1.1), fontsize=11, color='darkred', weight='bold') 
	ax3.set_ylim(0., 1.28)
	ax3.set_xlim(3700/10000, 18500./10000)
	ax3.set_title('2MASS', weight='bold')	
	plt.tick_params(left = False)
	plt.xlabel('$\lambda$ ($\mu$m)')
	plt.show()

###
def histo_periods():

	templates_sample = [X for X in M33.keys() if M33[X]['flag_pm11'] == '3' and M33[X]['sigma_phase']<0.05]
	HST_sample = [X for X in HST.keys()]

	plt.hist([np.log10(M33[X]['period']) for X in templates_sample], bins=int(len(M33)/28), color='deepskyblue', alpha=0.5, label='Template sample (N=%s)'%len(templates_sample))
	plt.hist([np.log10(M33[X]['period']) for X in HST_sample],       bins=int(len(M33)/22), color='magenta', alpha=0.5, label='HST sample (N=%s)'%len(HST_sample))

	plt.xlabel('logP (days)')
	plt.ylabel('N')
	plt.legend()
	plt.show()

###
def full_data_table():

	R = res

	for i in range(len(R)):

		if R['cep'][i] not in outliers:
			### Flag Cepheids in clusters to add asterisk:
			flag = ''
			if R['cep'][i] in clusters_ceps:
				flag = '$^{(*)}$'

			### Get SH0ES color:
			color_phatter = ufloat(R['V'][i], R['eV'][i]) - ufloat(R['I'][i], R['eI'][i])
			mh = ufloat(R['H'][i], R['eH'][i]) - 0.386*(0.065 + 0.658*(color_phatter))
			mhw, emhw = mh.nominal_value, mh.std_dev

			### Sample name:
			if (R['sample'][i] == 'go') or (R['cep'][i] in gold_sample_extended) or (R['sample'][i] in ['si', 'sg']):
				samp_name = 'G'
			else:
				samp_name = 'S'

			### Geometry correction and galactocentric distance:
			delta_corr = M33[R['cep'][i]]['delta_mag']
			d_kpc = M33[R['cep'][i]]['d_kpc']

			# print('%s %s & %.5f & %.5f & %.3f & $%.3f_{\pm %.3f}$ & $%.3f_{\pm %.3f}$ & $%.3f_{\pm %.3f}$ & $%.3f_{\pm %.3f}$ & %.2f & %.3f & %s \\\\'
			# 	%(R['cep'][i], flag, M33[R['cep'][i]]['RA'], M33[R['cep'][i]]['DEC'], np.log10(R['period'][i]), R['H'][i], R['eH'][i], R['V'][i], R['eV'][i], R['I'][i], R['eI'][i], mhw, emhw, d_kpc, delta_corr, samp_name))

			print('%s %s & %.5f & %.5f & %.3f & $%.3f_{\,(%.0f)}$ & $%.3f_{\,(%.0f)}$ & $%.3f_{\,(%.0f)}$ & $%.3f_{\,(%.0f)}$ & %.2f & %.3f & %s \\\\'
				%(R['cep'][i], flag, M33[R['cep'][i]]['RA'], M33[R['cep'][i]]['DEC'], np.log10(R['period'][i]), R['H'][i], R['eH'][i]*1000, R['V'][i], R['eV'][i]*1000, R['I'][i], R['eI'][i]*1000, mhw, emhw*1000, d_kpc, delta_corr, samp_name))


### Use this function for optical Wesenheit PL relations:
def PL_wvi():

	indices = [i for i in range(len(res)) if (res['sample'][i] in ['go', 'si', 'sg', 'br']) and (res['cep'][i] not in outliers)]
	cep           = [res['cep'][i]                                                         for i in indices]
	logP          = [np.log10(res['period'][i])                                            for i in indices]
	color_phatter = [ufloat(res['V'][i], res['eV'][i]) - ufloat(res['I'][i], res['eI'][i]) for i in indices]
	color_sh0es   = [0.065 + 0.658*col for col in color_phatter]
	Imag          = [res['I'][i]                                                           for i in indices]
	eImag         = [res['eI'][i]                                                          for i in indices]
	samp_name     = [res['sample'][i]                                                      for i in indices]
	delta_mag     = [M33[res['cep'][i]]['delta_mag']                                       for i in indices]

	### WH from HST/WFC3 magnitudes, random-phase corrected:	
	wvi = [ufloat(Imag[i], eImag[i]) + delta_mag[i] - 1.3*color_sh0es[i] for i in range(len(indices))]
	mw, emw = [y.nominal_value for y in wvi], [y.std_dev for y in wvi]
	emw = [np.sqrt(ey**2 + 0.07**2) for ey in emw]

	### Free slope:
	popt, pcov = curve_fit(lambda X, A, B: A*X+B, logP, mw, sigma=emw)
	perr = np.sqrt(np.diag(pcov))
	slope, zp = popt[0], popt[1]
	e_slope, e_zp = perr[0], perr[1]
	sigma = np.sqrt(sum( (np.array(mw)-np.array([slope*XX + zp for XX in logP]))**2 )/len(indices))
	fit = np.array([slope*XX + zp for XX in logP])

	chi2 = sum([(fit[i] - mw[i])**2/(emw[i]**2) for i in range(len(indices))])  
	chi2r = chi2/(len(indices)-2)

	### Fixed slope (R19):
	popt_fix, pcov_fix = curve_fit(lambda X, B: -3.31*X+B, logP, mw, sigma=emw)
	perr_fix = np.sqrt(np.diag(pcov_fix))
	slope_fix, zp_fix = -3.31, popt_fix[0]
	e_slope_fix, e_zp_fix = 0, perr_fix[0]

	print('\n Breuval+23: ')
	print(' Free slope:  W_VI = (%.3f ± %.3f) logP + (%.3f ± %.3f) '%(slope, e_slope, zp, e_zp))
	print(' Fixed slope: W_VI = (%.3f ± %.3f) logP + (%.3f ± %.3f) '%(-3.31, 0., zp_fix, e_zp_fix))
	print('\n sigma = %.3f mag    chi2r = %.3f    N=%i   \n'%(sigma, chi2r, len(indices)))

	
	plt.figure(figsize=(10,5))
	plt.subplots_adjust(left=0.06, right=0.99, top=0.98, bottom=0.10, hspace=0.1, wspace=0.3)

	plt.plot(xfine, slope_fix*(xfine)+zp_fix,   '-', linewidth=1.2, color='darkblue', label='Breuval+23: $m_{VI}^W = %.2f \, \log P + %.3f_{\pm %.3f}$ ($\sigma = %.2f$)'%(slope_fix, zp_fix, e_zp_fix, sigma))
	plt.plot(xfine, slope_abby*(xfine)+zp_abby, '-', linewidth=1.2, color='orange',   label='Lee+22:       $m_{VI}^W = %.2f \, \log P + %.3f_{\pm %.3f}$ ($\sigma = %.2f$)'%(slope_abby, zp_abby, e_zp_abby, sigma_abby))

	plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]=='br'], [mw[i] for i in range(len(cep)) if samp_name[i]=='br'], 
		yerr=[emw[i] for i in range(len(cep)) if samp_name[i]=='br'], fmt='o', color='lightgray', markerfacecolor='lightgray', 
		alpha=1., markeredgewidth=0.5, markersize=5, capsize=0, ecolor='lightgray', elinewidth=0.7, label='Silver sample')
	
	plt.errorbar([logP[i] for i in range(len(cep)) if samp_name[i]!='br'], [mw[i] for i in range(len(cep)) if samp_name[i]!='br'], 
		yerr=[emw[i] for i in range(len(cep)) if samp_name[i]!='br'], fmt='o', color='k', markerfacecolor='darkblue', 
		markeredgewidth=0.5, markersize=5, capsize=0, ecolor='darkblue', elinewidth=0.7, label='Gold sample')

	plt.xlabel('$\log P$ (days)', fontsize=11)
	plt.ylabel('$m_{VI}^W$ (mag)', fontsize=11)
	plt.gca().invert_yaxis()
	plt.xlim(0.45, 1.95)
	plt.ylim(21., 15.3)
	plt.legend(fontsize=11, loc='upper left', fancybox=True, shadow=True)
	plt.show()

	return(-3.31, 0., zp_abby, e_zp_abby)


### 
def distance_from_Wvi(zp_wvi, e_zp_wvi):

	### PL relation in the LMC (Riess+2019):
	zp_LMC, e_zp_LMC = 15.935, 0.010
	### Metallicity effect (Riess+2022):
	gamma = ufloat(-0.217, 0.046)
	### Metallicity of LMC and (mean) M33 from Bresolin+2011 gradient:
	OH_LMC = ufloat(-0.32, 0.01)
	OH_M33_sample_uf = [ufloat(-0.045, 0.006)*M33[X]['d_galac'] + ufloat(8.50, 0.02) - 8.69 for X in HST.keys()]
	OH_M33 = np.mean([o.nominal_value for o in OH_M33_sample_uf])
	
	### Derive M33 distance modulus:
	distance_modulus = LMC_DM + ufloat(zp_wvi, e_zp_wvi) - ufloat(zp_LMC, e_zp_LMC) - gamma*(ufloat(OH_M33, 0.03) - OH_LMC)
	DM_M33, e_DM_M33 = distance_modulus.nominal_value, distance_modulus.std_dev

	# distance_kpc = 10**((distance_modulus-10)/5)
	# dist_kpc, e_dist_kpc = distance_kpc.nominal_value, distance_kpc.std_dev

	print('\n New M33 distance (Wvi): %.3f ± %.3f mag'%(DM_M33, e_DM_M33))



###
def M33_reddening():

	ceps = [X for X in M33.keys() if M33[X]['Imag'][0] < 40]

	V_I_obs = [M33[X]['Vmag'][0] - M33[X]['Imag'][0] for X in ceps]
	V_I_int = [0.25*M33[X]['logP'] + 0.50 for X in ceps]

	E_B_V = [(V_I_obs[i] - V_I_int[i])/1.237 for i in range(len(ceps))]

	plt.hist(E_B_V, bins=20, color='blue', alpha=0.4)
	plt.title('M33 mean $E(B-V) = %.3f$ mag ($\sigma = %.3f$ mag)'%(np.mean(E_B_V), np.std(E_B_V)))
	plt.xlabel('$E(B-V)$ (mag)')
	plt.ylabel('N')
	plt.show()



###
