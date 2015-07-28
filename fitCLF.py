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


def CLF(mag,L_bins,dL,Mhost_min,Mhost_max,Mhost):
    """
    Calculate the galaxy conditional luminosity function (CLF).
    
    Parameters
    ----------
    mag: array_like
        absolute magnitude of galaxies
    
    L_bins: array_like, optional
        bin edges to use for the luminosity function

    dL: int
        bin width

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

    mag_bin = np.asarray([mag[i] for i in range(len(mag)) if  Mhost_min<log(Mhost[i],10)<=Mhost_max]) 
    Nhosts = len(mag_bin)
    print "len("+str((Mhost_max+Mhost_min)/2)+")"
    print Nhosts
    Msun = 4.76 # The Sun's absolute magnitude
    L = ((Msun-mag_bin)/2.5) # Calculated the luminosity 
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
        the length of one dimension of the full box.
        
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



def compute_CLF_jackknife(L_bins,dL,M_cen,Mhost_min,Mhost_max,Mhost_cen):
	M = 10**((Mhost_max+Mhost_min)/2)
	print "calculating clf for the full sample"
	Phi_cen,L_bins = CLF(M_cen,L_bins,dL,Mhost_min,Mhost_max,Mhost_cen)
	print "finished calculating clf for the full sample"
	errors = jackknife_errors_CLF(pos_cen,Phi_cen,Ndivs,Lbox,M_cen,L_bins,dL,Mhost_min,Mhost_max,Mhost_cen)
	y_cen = [log(i,10) for i in Phi_cen if i>0]
	x_cen = [L_bins[i]+dL/2 for i in range(len(L_bins)-1) if Phi_cen[i]>0]
	delta_y_cen = [errors[i] for i in range(len(L_bins)-1) if Phi_cen[i]>0]
	yerr_cen = [i/(10**j) for i,j in zip(delta_y_cen,y_cen)]
	
	return x_cen,y_cen,yerr_cen 
	
	
def compute_CLF_jackknife_sat(L_bins,dL,M_sat,Mhost_min,Mhost_max,Mhost_sat):
	M = 10**((Mhost_max+Mhost_min)/2)
	Phi_sat,L_bins = CLF(M_sat,L_bins,dL,Mhost_min,Mhost_max,Mhost_sat)
	errors = jackknife_errors_CLF(pos_sat,Phi_sat,Ndivs,Lbox,M_sat,L_bins,dL,Mhost_min,Mhost_max,Mhost_sat)
	y_sat = [log(i,10) for i in Phi_sat if i>0]
	x_sat = [L_bins[i]+dL/2 for i in range(len(L_bins)-1) if Phi_sat[i]>0]
	delta_y_sat = [errors[i] for i in range(len(L_bins)-1) if Phi_sat[i]>0]
	yerr_sat = [i/(10**j) for i,j in zip(delta_y_sat,y_sat)]	
	return x_sat,y_sat,yerr_sat



def plot_clf_fit_cen(x_cen,y_cen, yerr,M_host_min,M_host_max):
	label = str(M_host_min) +r'$ < M_{h}  \leq $' + str(M_host_max)
	mid_halo_mass = (M_host_min+M_host_max)/2
	plt.errorbar(x_cen,y_cen, yerr=yerr, capsize=4, ls='none', color='red', elinewidth=2,marker='o',markerfacecolor='red')
	x_fit_cen = np.arange(x_cen[-len(x_cen)]-0.1,x_cen[-1]+0.1,0.05)
	y_fit_plot_cen = [log(n,10) for n in peval(x_fit_cen, fitted_para_cen,10**mid_halo_mass)]
	plt.plot(x_fit_cen,y_fit_plot_cen,color='black')
	plt.xlim(7.5, 12)
	plt.xlabel(r'$\log[L/(h^{-2}L_{\bigodot})])$',fontsize=15)
	plt.ylabel(r'$\log(\Phi(L) d\log L / group)$',fontsize=15)
	plt.title(label)
	plt.savefig('plots/clf_cen_fit_'+str(M_host_min)+'-'+str(M_host_max)+'.png')
	"""
	if (M_host_min==11.75):
		print "I'm inside plot_clf_fit_cen:"
		print "mid_halo_mass:"
		print mid_halo_mass
		print "x_fit_cen:"
		print x_fit_cen
		print "peval(x_fit_cen, fitted_para_cen,10**mid_halo_mass):"
		print peval(x_fit_cen, fitted_para_cen,10**mid_halo_mass)
		print "y_fit_plot_cen:"
		print y_fit_plot_cen
		plt.plot(x_fit_cen,y_fit_plot_cen,color='black')
		plt.show()
		"""
	plt.close() 


def plot_clf_fit_sat(x_sat,y_sat, yerr,M_host_min,M_host_max):
	label = str(M_host_min) +r'$ < M_{h}  \leq $' + str(M_host_max)
	mid_halo_mass = (M_host_min+M_host_max)/2
	plt.errorbar(x_sat,y_sat, yerr=yerr, capsize=4, ls='none', color='red', elinewidth=2,marker='o',markerfacecolor='red')
	x_fit_sat = np.arange(x_sat[-len(x_sat)]-0.1,x_sat[-1]+0.1,0.05)
	y_fit_plot_sat = [log(n,10) for n in peval_sat(x_fit_sat, fitted_para_sat,fitted_para_cen,10**mid_halo_mass)]
	plt.plot(x_fit_sat,y_fit_plot_sat,color='black')
	plt.xlim(7.5, 12)
	plt.xlabel(r'$\log[L/(h^{-2}L_{\bigodot})])$',fontsize=15)
	plt.ylabel(r'$\log(\Phi(L) d\log L / group)$',fontsize=15)
	plt.title(label)
	plt.savefig('plots/clf_sat_fit_'+str(M_host_min)+'-'+str(M_host_max)+'.png')
	plt.close() 




"""
Main program
"""


# General useful quantities
Lbox = 250.0
Vbox = Lbox**3
Ndivs = 2


"""
Data processing 
"""

# Reads the data from the mock 
data1 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/erased_assembly_bias_Mr_gr_model.dat",float) # no assembly bias
#data1 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/conditional_abundance_matching_Mr_gr_model.dat",float) # with assembly bias
M = np.asarray(data1[:,7]) # The absoulute magnitude
pos = data1[:,range(1,4)] # the galaxies positions
N = len(M) # the number of data points (galaxies)

data2 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/noab_mock.dat",float) # no assembly bias
#data2 = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Data/sham_mock.dat",float) # with assembly bias
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
L_min = 9.7	
L_max = 11.5		
dL = 0.15
#print "(L_max-L_min)/dL:"
#print (L_max-L_min)/dL
L_bins = np.linspace(L_min,L_max,(L_max-L_min)/dL+1)
#L_bins = np.linspace(L_min,L_max,(L_max-L_min)/dL)
#print "L_bins:"
#print L_bins


"""
Calculate the CLF for central galaxies and its Jackknife errors for 10 different halo mass bins
"""
# Shorten this set of commands
print "halo mass bin #0"
x_cen_0,y_cen_0,yerr_cen_0 = compute_CLF_jackknife(L_bins,dL,M_cen,11.75,12.0,Mhost_cen)
print "\n"
print "halo mass bin #1"
x_cen_1,y_cen_1,yerr_cen_1 = compute_CLF_jackknife(L_bins,dL,M_cen,12.0,12.25,Mhost_cen)
print "\n"
print "halo mass bin #2"
x_cen_2,y_cen_2,yerr_cen_2 = compute_CLF_jackknife(L_bins,dL,M_cen,12.25,12.5,Mhost_cen)
print "\n"
print "halo mass bin #3"
x_cen_3,y_cen_3,yerr_cen_3 = compute_CLF_jackknife(L_bins,dL,M_cen,12.5,12.75,Mhost_cen)
print "\n"
print "halo mass bin #4"
x_cen_4,y_cen_4,yerr_cen_4 = compute_CLF_jackknife(L_bins,dL,M_cen,12.75,13.0,Mhost_cen)
print "\n"
print "halo mass bin #5"
x_cen_5,y_cen_5,yerr_cen_5 = compute_CLF_jackknife(L_bins,dL,M_cen,13.0,13.25,Mhost_cen)
print "\n"
print "halo mass bin #6"
x_cen_6,y_cen_6,yerr_cen_6 = compute_CLF_jackknife(L_bins,dL,M_cen,13.25,13.5,Mhost_cen)
print "\n"
print "halo mass bin #7"
x_cen_7,y_cen_7,yerr_cen_7 = compute_CLF_jackknife(L_bins,dL,M_cen,13.5,13.75,Mhost_cen)
print "\n"
print "halo mass bin #8"
x_cen_8,y_cen_8,yerr_cen_8 = compute_CLF_jackknife(L_bins,dL,M_cen,13.75,14.0,Mhost_cen)
print "\n"
print "halo mass bin #9"
x_cen_9,y_cen_9,yerr_cen_9 = compute_CLF_jackknife(L_bins,dL,M_cen,14.0,14.25,Mhost_cen)
print "\n"


x_cen = x_cen_0 + x_cen_1 + x_cen_2 + x_cen_3 + x_cen_4 +x_cen_5 + x_cen_6 + x_cen_7 +x_cen_8 + x_cen_9 
y_cen =  y_cen_0 + y_cen_1 + y_cen_2 + y_cen_3 + y_cen_4 +y_cen_5 + y_cen_6 + y_cen_7 +y_cen_8 + y_cen_9 
yerr_cen = yerr_cen_0 + yerr_cen_1 + yerr_cen_2 + yerr_cen_3 + yerr_cen_4 + yerr_cen_5 + yerr_cen_6 +yerr_cen_7 +yerr_cen_8 + yerr_cen_9 
Mass_bins = [10**11.875 for i in range(len(x_cen_0))] + [10**12.125 for i in range(len(x_cen_1))] + [10**12.375 for i in range(len(x_cen_2))] + [10**12.625 for i in range(len(x_cen_3))] +  [10**12.875 for i in range(len(x_cen_4))] + [10**13.125 for i in range(len(x_cen_5))]  + [10**13.375 for i in range(len(x_cen_6))]+[10**13.625 for i in range(len(x_cen_7))]+[10**13.875 for i in range(len(x_cen_8))]+[10**14.125 for i in range(len(x_cen_9))] 


"""
"Data points for centrals:"
print len(x_cen)
print len(y_cen)
print len(yerr_cen)
print len(Mass_bins)
"""

"""
Fit the CLF model - Central Galaxies
"""


def peval(x_lum,p,M):
	x_lum = np.asarray(x_lum)
	x_lc = p[0]+ p[2]*(log(M,10)-p[1])-(p[2]-p[3])*log(1+(M/(10**p[1])),10)
	clf = (1/(sqrt(2*pi)*p[4])*np.exp(-((x_lum-x_lc)/(sqrt(2)*p[4]))**2)) 
	return clf

def residuals_cen(p,y,x_lum,errors,M):
	x_lum = np.asarray(x_lum)
	y = np.asarray(y)
	M = np.asarray(M)
	logL0,logM1,gamma1,gamma2,sigma_c = p
	x_lc = logL0+ gamma1*(np.log10(M)-logM1)-(gamma1-gamma2)*np.log10(1+(M/(10**logM1)))
	clf = (1/(sqrt(2*pi)*sigma_c)*np.exp(-((x_lum-x_lc)/(sqrt(2)*sigma_c))**2)) 
	delta = 10**-10
	err = (np.asarray([log(k+delta,10) for k in y]) - np.asarray([log(p+delta,10) for p in clf]))/errors
	return err

p0 = [9.9,11.0,3.0,0.3,0.15] # Initial values for [logL0,logM1,gamma1,gamma2,sigma_c]

y = [10**i for i in y_cen]


plsq = leastsq(residuals_cen, p0, args=(y, x_cen,yerr_cen,Mass_bins))
print "The parameters from the fit for the centrals galaxies are:"
print plsq[0]
print "\n"
fitted_para_cen = plsq[0]



# Calculating Chi Square

x_lc = fitted_para_cen[0]+ fitted_para_cen[2]*(np.log10(Mass_bins)-fitted_para_cen[1])-(fitted_para_cen[2]-fitted_para_cen[3])*np.log10(1+(Mass_bins/(10**fitted_para_cen[1])))
clf_cen = (1/(sqrt(2*pi)*fitted_para_cen[4])*np.exp(-((x_cen-x_lc)/(sqrt(2)*fitted_para_cen[4]))**2))


y_observed = np.asarray(y_cen)
y_expected = np.asarray([log(i,10) for i in clf_cen])
#y_expected = np.asarray(peval(x_cen, plsq[0],M))
#y_expected = np.asarray([log(i,10) for i in y_expected])
sigma2 = np.asarray(yerr_cen)**2
chi2_cen = np.sum((y_observed-y_expected)**2/sigma2)/(len(y_observed)-6)
print "chi2 for central galaxies CLF fit:"
print chi2_cen
print "\n"


"""
print "x_cen_0:"
print x_cen_0
print "y_cen_0:"
print y_cen_0
print "for mass bin 11.75 - 12.0, mid point 11.875"
"""


plot_clf_fit_cen(x_cen_0,y_cen_0, yerr_cen_0,11.75,12.0)
plot_clf_fit_cen(x_cen_1,y_cen_1, yerr_cen_1,12.0,12.25)
plot_clf_fit_cen(x_cen_2,y_cen_2, yerr_cen_2,12.25,12.5)
plot_clf_fit_cen(x_cen_3,y_cen_3, yerr_cen_3,12.5,12.75)
plot_clf_fit_cen(x_cen_4,y_cen_4, yerr_cen_4,12.75,13.0)
plot_clf_fit_cen(x_cen_5,y_cen_5, yerr_cen_5,13.0,13.25)
plot_clf_fit_cen(x_cen_6,y_cen_6, yerr_cen_6,13.25,13.5)
plot_clf_fit_cen(x_cen_7,y_cen_7, yerr_cen_7,13.5,13.75)
plot_clf_fit_cen(x_cen_8,y_cen_8, yerr_cen_8,13.75,14.0)
plot_clf_fit_cen(x_cen_9,y_cen_9, yerr_cen_9,14.0,14.25)


print "********************************************** Finished centrals!!! **********************************************"



print "********************************************** Starting satellites!!! **********************************************"

"""
Calculate the CLF for satellite galaxies and its Jackknife errors for 9 different halo mass bins
"""
# Shorten this set of commands
print "halo mass bin #0"
x_sat_0,y_sat_0,yerr_sat_0 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,11.75,12.0,Mhost_sat)
print "\n"
print "halo mass bin #1"
x_sat_1,y_sat_1,yerr_sat_1 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,12.0,12.25,Mhost_sat)
print "\n"
print "halo mass bin #2"
x_sat_2,y_sat_2,yerr_sat_2 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,12.25,12.5,Mhost_sat)
print "\n"
print "halo mass bin #3"
x_sat_3,y_sat_3,yerr_sat_3 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,12.5,12.75,Mhost_sat)
print "\n"
print "halo mass bin #4"
x_sat_4,y_sat_4,yerr_sat_4 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,12.75,13.0,Mhost_sat)
print "\n"
print "halo mass bin #5"
x_sat_5,y_sat_5,yerr_sat_5 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,13.0,13.25,Mhost_sat)
print "\n"
print "halo mass bin #6"
x_sat_6,y_sat_6,yerr_sat_6 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,13.25,13.5,Mhost_sat)
print "\n"
print "halo mass bin #7"
x_sat_7,y_sat_7,yerr_sat_7 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,13.5,13.75,Mhost_sat)
print "\n"
print "halo mass bin #8"
x_sat_8,y_sat_8,yerr_sat_8 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,13.75,14.0,Mhost_sat)
print "\n"
print "halo mass bin #9"
x_sat_9,y_sat_9,yerr_sat_9 = compute_CLF_jackknife_sat(L_bins,dL,M_sat,14.0,14.25,Mhost_sat)
print "\n"




x_sat = x_sat_0 + x_sat_1 + x_sat_2 + x_sat_3 + x_sat_4 +x_sat_5 + x_sat_6 + x_sat_7 +x_sat_8 + x_sat_9 
y_sat = y_sat_0 + y_sat_1 + y_sat_2 + y_sat_3 + y_sat_4 +y_sat_5 + y_sat_6 + y_sat_7 +y_sat_8 + y_sat_9 
yerr_sat = yerr_sat_0 + yerr_sat_1 + yerr_sat_2 + yerr_sat_3 + yerr_sat_4 + yerr_sat_5 + yerr_sat_6 + yerr_sat_7 + yerr_sat_8 + yerr_sat_9 
Mass_bins_sat = [10**11.875 for i in range(len(x_sat_0))] + [10**12.125 for i in range(len(x_sat_1))] + [10**12.375 for i in range(len(x_sat_2))] + [10**12.625 for i in range(len(x_sat_3))] +  [10**12.875 for i in range(len(x_sat_4))] + [10**13.125 for i in range(len(x_sat_5))]  + [10**13.375 for i in range(len(x_sat_6))]+[10**13.625 for i in range(len(x_sat_7))]+[10**13.875 for i in range(len(x_sat_8))]+[10**14.125 for i in range(len(x_sat_9))] 

"""
print "Data points for satellites:"
print len(x_sat)
print len(y_sat)
print len(yerr_sat)
print len(Mass_bins)
print "\n"
"""

"""
Fit the CLF model - Satellite Galaxies
"""


x_lc = fitted_para_cen[0]+ fitted_para_cen[2]*(np.log10(np.asarray(Mass_bins_sat))-fitted_para_cen[1])-(fitted_para_cen[2]-fitted_para_cen[3])*np.log10(1+(np.asarray(Mass_bins_sat)/(10**fitted_para_cen[1])))
Lc = 10**x_lc
Ls = 0.562*Lc 


def peval_sat(x_lum,p_sat,p_cen,M): # still need to modify
	logM12 = log(M,10)-12.0
	phi_s = 10**(p_sat[0]+p_sat[1]*logM12+p_sat[2]*logM12**2)
	x_lc = p_cen[0]+ p_cen[2]*(np.log10(np.asarray(M))-p_cen[1])-(p_cen[2]-p_cen[3])*np.log10(1+(np.asarray(M)/(10**p_cen[1])))
	Lc = 10**x_lc
#	Ls = 0.562*Lc 
	Ls = p_sat[5]*Lc 
	Lrat = 10**np.asarray(x_lum)/Ls
#	return log(10,e)*phi_s*Lrat**(p_sat[3]+1)*np.exp(-Lrat**2)
	return log(10,e)*phi_s*Lrat**(p_sat[3]+1)*np.exp(-Lrat**p_sat[4])
#	return log(10,e)*phi_s*Lrat**(p_sat[3]+1)*np.exp(-Lrat**1.28)



# Fitting the CLF for satellite galaxies
def residuals_sat(p,y,x_lum,errors,M):
	b0,b1,b2,alpha_sat,power,f = p
	logM12 = np.log10(M)-12.0
	phi_s = 10**(b0+b1*logM12+b2*logM12**2)
	Ls = f*Lc
	Lrat = 10**np.asarray(x_lum)/np.asarray(Ls)
	#x_lum = np.asarray(x_lum)
	y = np.asarray(y)
#	clf = log(10,e)*phi_s*(Lrat**(alpha_sat+1))*np.exp(-Lrat**2)
	clf = log(10,e)*phi_s*(Lrat**(alpha_sat+1))*np.exp(-Lrat**power)
#	clf = log(10,e)*phi_s*(Lrat**(alpha_sat+1))*np.exp(-Lrat**1.28)

	err = (np.asarray([log(k,10) for k in y]) - np.asarray([log(p,10) for p in clf]))/errors
	#err = np.asarray([log(k+delta,10) for k in y]) - np.asarray([log(p+delta,10) for p in clf])
	return err




p0_sat = [-1.0,2.0,-0.5,-0.5,1.5,0.5] # Initial values for [b0,b1,b2,alpha_sat, power, f]

y = [10**j for j in y_sat]

plsq_sat = leastsq(residuals_sat, p0_sat, args=(y, x_sat,yerr_sat,Mass_bins_sat))
print "The parameters from the fit for the satellite galaxies:"
print plsq_sat[0]
fitted_para_sat = plsq_sat[0]
print "\n"



# Calculating Chi Square

logM12 = np.log10(Mass_bins_sat)-12.0
phi_s = 10**(fitted_para_sat[0]+fitted_para_sat[1]*logM12+fitted_para_sat[2]*logM12**2)
Ls = fitted_para_sat[5]*Lc
Lrat = 10**np.asarray(x_sat)/np.asarray(Ls)
#clf_sat = log(10,e)*phi_s*(Lrat**(fitted_para_sat[3]+1))*np.exp(-Lrat**2)
clf_sat = log(10,e)*phi_s*(Lrat**(fitted_para_sat[3]+1))*np.exp(-Lrat**fitted_para_sat[4])


y_observed = np.asarray(y_sat)
y_expected = np.asarray([log(i,10) for i in clf_sat])
sigma2 = np.asarray(yerr_sat)**2
chi2_sat = np.sum((y_observed-y_expected)**2/sigma2)/(len(y_observed)-6)
print "chi2 for satellite galaxies CLF fit:"
print chi2_sat
print "\n"


plot_clf_fit_sat(x_sat_0,y_sat_0, yerr_sat_0,11.75,12.0)
plot_clf_fit_sat(x_sat_1,y_sat_1, yerr_sat_1,12.0,12.25)
plot_clf_fit_sat(x_sat_2,y_sat_2, yerr_sat_2,12.25,12.5)
plot_clf_fit_sat(x_sat_3,y_sat_3, yerr_sat_3,12.5,12.75)
plot_clf_fit_sat(x_sat_4,y_sat_4, yerr_sat_4,12.75,13.0)
plot_clf_fit_sat(x_sat_5,y_sat_5, yerr_sat_5,13.0,13.25)
plot_clf_fit_sat(x_sat_6,y_sat_6, yerr_sat_6,13.25,13.5)
plot_clf_fit_sat(x_sat_7,y_sat_7, yerr_sat_7,13.5,13.75)
plot_clf_fit_sat(x_sat_8,y_sat_8, yerr_sat_8,13.75,14.0)
plot_clf_fit_sat(x_sat_9,y_sat_9, yerr_sat_9,14.0,14.25)


print "********************************************** Finished satellites!!! **********************************************"



"""
num_bins = 300

# Calculates the CLF using the parameters from the fit and saves it into dat file
# We will use it inside a Riemann sum for calculating the Luminosity function
#mass_mid_points = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Code/mass_function/mass_mid_bins.dat",float)
mass_mid_points = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Code/mass_function/MF_Code_Tinker/tinker_"+str(num_bins)+"_bins.dndM",float)[:,0]
mass_mid_points = [log10(i) for i in mass_mid_points]
x = np.loadtxt("/Users/si224/Documents/2014/vanDenBosch/Code/luminosityFunction/Output/direct_calc.dat",float)
L = x[:,0]

L_no_log = np.asarray([10**x for x in L])

#clf_cen = np.asarray([peval(L, fitted_para_cen,10**x) for x in mass_mid_points])
#clf_sat = np.asarray([peval_sat(L, fitted_para_sat,fitted_para_cen,10**x) for x in mass_mid_points])
clf_cen = np.asarray([peval(L, fitted_para_cen,10**x) for x in mass_mid_points])
clf_sat = np.asarray([peval_sat(L, fitted_para_sat,fitted_para_cen,10**x) for x in mass_mid_points])



np.savetxt("output/L.dat",L)

#x_cen = clf_cen.reshape((24,20))
#x_cen = clf_cen.reshape((399,20))
x_cen = clf_cen.reshape((num_bins,20))
Phi_cen = x_cen/(L_no_log*log(10,e))
#np.savetxt("output/clf_cen.dat", Phi_cen)
np.savetxt("output/clf_cen.dat", x_cen)

#x_sat = clf_sat.reshape((24,20))
#x_sat = clf_sat.reshape((399,20))
x_sat = clf_sat.reshape((num_bins,20))
Phi_sat = x_sat/(L_no_log*log(10,e))
#np.savetxt("output/clf_sat.dat", Phi_sat)
np.savetxt("output/clf_sat.dat", x_sat)
"""























