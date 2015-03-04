#!/usr/bin/python

# Shany Danieli
# February 27, 2015
# Yale University

""" 
The program calculates the CLF and its Jackknife errors.
The program fits the CLF model.
"""


####import modules########################################################################
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import random
##########################################################################################


def CLF(M,L_bins,dL,Mhostmin,Mhost_max,Mhost):
    """
    Calculate the galaxy conditional luminosity function (CLF).
    
    Parameters
    ----------
    M: array_like
        absolute magnitude of galaxies
    
    L_bins: array_like, optional
        bin edges to use for for the luminosity function

    dL: int
        bin width

    Nhosts: int
        the number of host halos

    Mhalo_min: int, log (Mhalo)
        the lower limit of the host halo mass bin 

    Mhalo_max: int, log (Mhalo)
        the upper limit of the host halo mass bin

    Mhost: array_like
        the mass of the host halos

    
    Returns
    -------
    CLF, L_bins: np.array, np.array
    """

    M_bin = np.asarray([M[i] for i in range(len(M)) if  Mhost_min<log(Mhost[i],10)<=Mhost_max]) # First luminosity bin
    Nhosts = len(M_bin)
    Msun = 4.76
    L = ((Msun-M_bin)/2.5)
    counts = np.histogram(L,L_bins)[0]
    CLF = counts/(dL*Nhosts)
    return CLF,L_bins



def jackknife_errors_CLF(pos,Phi,Ndivs,Lbox,M,L_bins,dL,Mhost_min,Mhost_max,Mhost):
    """
    Calculate the errors to the CLF using Jackknife resampling method
    
    Parameters
    ----------
    pos: array-like
        Npts x 3 numpy array containing 3d positions of Npts.
        
    Phi: array-like
        the CLF for the full box.

    Ndivs: int
        the number by which we divide every dimension in the Jackknife resampling.

    Lbox: int
        bla bla
        
    M: array-like
        the absolute magnitude of the galaxies.
        
    L_bins: array_like
    	numpy array of the left boundaries defining the luminosity bins in which pairs are counted. 
    
	dL: float
		the luminosity bin size.
		
	Mhost_min: float
		the lower bound for the halo mass bin.
		
	Mhost_max: float		
		the upper bound for the halo mass bin.
		
	Mhost: array-like
		Npts numpy array containing the halo masses of Npts.	

			
    Returns
    -------
    Jackknife errors for the different luminosity bins: np.array
    """

    n_subBox = Ndivs*Ndivs*Ndivs # The number of sub volumes for the Jackknife resampling
    V_subBox = Vbox - Vbox/n_subBox # The volume of a Jackknife sample
    N = len(pos) 
    delta = Lbox/Ndivs
    
    # Indices for the galaxies positions
    index = np.asarray([floor(pos[i,0]/delta) + (floor(pos[i,1]/delta)*Ndivs) + (floor(pos[i,2]/delta)*Ndivs*Ndivs) + 1 for i in range(N)]) # index for the position of particle2
    M_sub_sample = [] # keeps the absolute magnitude for the sub-samples
    Mhost_sub_sample = [] # keeps the halo mass for the sub-samples
    CLF_all = []  # keeps the values of the CLF for the full sample and for each of the sub-samples
    CLF_all.append(Phi)
    for k in range(1,n_subBox+1): # run over the sub-samples
        for i in range(0,N): # runs over all the points (galaxies)
                if (index[i] != k): # the point is inside the sub-box
                    M_sub_sample.append(M[i]) # then add to sub-box list
                    Mhost_sub_sample.append(Mhost[i])
        CLF_sub,L_bins = CLF(M_sub_sample,L_bins,dL,Mhost_min,Mhost_max,Mhost_sub_sample)
        CLF_all.append(CLF_sub)
        M_sub_sample = []
        Mhost_sub_sample = []

	n_subBox = float(n_subBox)
    full = np.asarray(CLF_all[0]) # the CLF for the full sample
    sub_samples = np.asarray(CLF_all[1:]) # the CLF for the Jackknife sub-samples
    after_subtraction =  sub_samples - np.mean(sub_samples,axis=0)
    squared = after_subtraction**2
    error2 = ((n_subBox-1)/n_subBox)*squared.sum(axis=0)
    errors = error2**0.5
    return errors



# General useful quantities
Lbox = 250.0
Vbox = Lbox**3
Ndivs = 5

"""
Data processing 
"""

# Reads the data from the mock 
data1 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/erased_assembly_bias_Mr_gr_model.dat",float)
#data1 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/conditional_abundance_matching_Mr_gr_model.dat",float)
M = np.asarray(data1[:,7]) # The absoulute magnitude
pos = data1[:,range(1,4)] # the galaxies positions
N = len(M)

data2 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/noab_mock.dat",float)
#data2 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/sham_mock.dat",float)
halo_ID = data2[:,0] # Halo ID
UPID = data2[:,1] # if UPID == -1 -> central, if UPID > 0 -> satellite
Mhost = data2[:,2] # Host mass in units of (Msun/h)
halo_ID = np.asarray(halo_ID)
UPID = np.asarray(UPID)
Mhost = np.asarray(Mhost)

# Split sample into central and satellite galaxies
M_cen = []
M_sat = []
Mhost_cen = []
Mhost_sat = []
pos_cen = []
pos_sat = []
for i in range(N):
    if UPID[i]<0:
        M_cen.append(M[i])
        Mhost_cen.append(Mhost[i])
        pos_cen.append(pos[i])
    else:
        M_sat.append(M[i])
        Mhost_sat.append(Mhost[i])
        pos_sat.append(pos[i])
pos_cen = np.asarray(pos_cen)
pos_sat = np.asarray(pos_sat)

# logaritmic bins for L
L_min = 8.5
L_max = 11.5		
dL = 0.08

L_bins = np.linspace(L_min,L_max,(L_max-L_min)/dL)

print "L_bins:"
print L_bins

# Getting host halo mass bin from user
#Mhost_min = input("Hello! Please enter the lower limit of the mass bin in logarithmic scale: ")
#Mhost_max = input("Now, enter the upper limit of the mass bin in logarithmic scale: ")
Mhost_min = 12.7
Mhost_max = 13.0
print "\n"
print "Plotting the Conditional Luminosity Function..."



"""
CLF and Jackknife computation 
"""

# Computing the CLF and its Jackknife errors - Central galaxies
Phi_cen,L_bins = CLF(M_cen,L_bins,dL,Mhost_min,Mhost_max,Mhost_cen)
errors = jackknife_errors_CLF(pos_cen,Phi_cen,Ndivs,Lbox,M_cen,L_bins,dL,Mhost_min,Mhost_max,Mhost_cen)
y_cen = [log(i,10) for i in Phi_cen if i>0]
x_cen = [L_bins[i]+dL/2 for i in range(len(L_bins)-1) if Phi_cen[i]>0]
delta_y_cen = [errors[i] for i in range(len(L_bins)-1) if Phi_cen[i]>0]
yerr_cen = [i/(10**j) for i,j in zip(delta_y_cen,y_cen)]

# Computing the CLF and its Jackknife errors - Satellite galaxies
Phi_sat,L_bins = CLF(M_sat,L_bins,dL,Mhost_min,Mhost_max,Mhost_sat)
errors = jackknife_errors_CLF(pos_sat,Phi_sat,Ndivs,Lbox,M_sat,L_bins,dL,Mhost_min,Mhost_max,Mhost_sat)
y_sat = [log(i,10) for i in Phi_sat if i>0]
x_sat = [L_bins[i]+dL/2 for i in range(len(L_bins)-1) if Phi_sat[i]>0]
delta_y_sat = [errors[i] for i in range(len(L_bins)-1) if Phi_sat[i]>0]
yerr_sat = [i/(10**j) for i,j in zip(delta_y_sat,y_sat)]
    


"""
Fit the CLF model - Central Galaxies
"""

def peval(x_lum,p):
	M = 10**12.85
	x_lum = np.asarray(x_lum)
	x_lc = p[0]+ p[2]*(log(M,10)-p[1])-(p[2]-p[3])*log(1+(M/(10**p[1])),10)
	return (1/(sqrt(2*pi)*p[4])*np.exp(-((x_lum-x_lc)/(sqrt(2)*p[4]))**2))

def residuals_c(p,y,x_lum,errors):
	M = 10**12.85
#    M = input("Enter the mid value of the host halo mass bin once more please: ")
	logL0,logM1,gamma1,gamma2,sigma_c = p
	x_lc = logL0+ gamma1*(log(M,10)-logM1)-(gamma1-gamma2)*log(1+(M/(10**logM1)),10)
	x_lum = np.asarray(x_lum)
	y = np.asarray(y)
	clf = (1/(sqrt(2*pi)*sigma_c)*np.exp(-((x_lum-x_lc)/(sqrt(2)*sigma_c))**2))
	delta = 10**-10
	err = (np.asarray([log(k+delta,10) for k in y]) - np.asarray([log(p+delta,10) for p in clf]))/errors
	return err

p0 = [9.9,11.0,3.0,0.3,0.15] # Initial values

y = [10**i for i in y_cen]
plsq = leastsq(residuals_c, p0, args=(y, x_cen,yerr_cen))
print "The parameters from the fit for the centrals galaxies are:"
print plsq[0]
print "\n"
fitted_para_cen = plsq[0]

# Calculating Chi Square
y_observed = np.asarray(y_cen)
y_expected = np.asarray(peval(x_cen, plsq[0]))
y_expected = np.asarray([log(i,10) for i in y_expected])
sigma2 = np.asarray(yerr_cen)**2
chi2 = np.sum((y_observed-y_expected)**2/sigma2)/(len(y_observed)-6)
print "chi2 for central galaxies CLF fit:"
print chi2


"""
Fit the CLF model - Satellite Galaxies
"""

M = 10**12.85
x_lc = fitted_para_cen[0]+ fitted_para_cen[2]*(log(M,10)-fitted_para_cen[1])-(fitted_para_cen[2]-fitted_para_cen[3])*log(1+(M/(10**fitted_para_cen[1])),10)
Lc = 10**x_lc
Ls = 0.562*Lc 


def peval_sat(x_lum,p): # still need to modify
	M = 10**12.85
	logM12 = log(M,10)-12.0
	phi_s = 10**(p[0]+p[1]*logM12+p[2]*logM12**2)
	Lrat = 10**np.asarray(x_lum)/np.asarray(Ls)
#    print log(10,e)*phi_s*Lrat**(p[3]+1)*np.exp(-Lrat**2)
	return log(10,e)*phi_s*Lrat**(p[3]+1)*np.exp(-Lrat**2)

# Fitting the CLF for central galaxies
def residuals_sat(p,y,x_lum,errors):
	M = 10**12.85
	b0,b1,b2,alpha_sat = p
	logM12 = log(M,10)-12.0
	phi_s = 10**(b0+b1*logM12+b2*logM12**2)
	Lrat = 10**np.asarray(x_lum)/np.asarray(Ls)
	#x_lum = np.asarray(x_lum)
	y = np.asarray(y)
	clf = log(10,e)*phi_s*(Lrat**(alpha_sat+1))*np.exp(-Lrat**2)
	err = (np.asarray([log(k,10) for k in y]) - np.asarray([log(p,10) for p in clf]))/errors
	#err = np.asarray([log(k+delta,10) for k in y]) - np.asarray([log(p+delta,10) for p in clf])
	return err


p0_sat = [-1.0,2.0,-0.5,-0.5] # Initial values

y = [10**j for j in y_sat]

plsq_sat = leastsq(residuals_sat, p0_sat, args=(y, x_sat,yerr_sat))
print "The parameters from the fit for the satellite galaxies:"
print plsq_sat[0]
print "\n"

# Calculating Chi Square
y_observed = np.asarray(y_sat)
y_expected = np.asarray(peval_sat(x_sat, plsq_sat[0]))
y_expected = np.asarray([log(i,10) for i in y_expected])
sigma2 = np.asarray(yerr_sat)**2
chi2 = np.sum((y_observed-y_expected)**2/sigma2)/(len(y_observed)-5)
print "chi2 for satellite galaxies CLF fit:"
print chi2



"""
Fit the CLF model - Satellite Galaxies
"""

label = str(Mhost_min)+r'$ < M_{h}  \leq $' + str(Mhost_max)

plt.errorbar(x_cen,y_cen, yerr=yerr_cen, capsize=4, ls='none', color='red', elinewidth=2,marker='o',markerfacecolor='none')
x_fit_cen = np.arange(x_cen[-len(x_cen)]-0.2,x_cen[-1]+0.2,0.05)
y_fit_plot_cen = [log(n,10) for n in peval(x_fit_cen, plsq[0])]
plt.plot(x_fit_cen,y_fit_plot_cen)



plt.errorbar(x_sat,y_sat, yerr=yerr_sat, capsize=4, ls='none', color='blue', elinewidth=2,marker='o',markerfacecolor='none')
x_fit_sat = np.arange(x_sat[-len(x_sat)]-0.05,x_sat[-1],0.05)
y_fit_plot_sat = [log(n,10) for n in peval_sat(x_fit_sat, plsq_sat[0])]
plt.plot(x_fit_sat,y_fit_plot_sat)


plt.xlim(8.5, 11.5)
plt.xlabel(r'$\log[L/(h^{-2}L_{\bigodot})])$',fontsize=15)
plt.ylabel(r'$\log(\Phi(L) d\log L / group)$',fontsize=15)
plt.title(label)
plt.show()











