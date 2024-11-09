import numpy as np
from math import sin, cos, pi, sqrt
from pylab import plot, show, xlabel, ylabel, legend
from numpy import zeros, empty, array, loadtxt, savetxt, copy

###############################################################################
# PS: This is my modified version of the code. This is what I used to generate
# the data using the original (old) paramaters on both SHM and SHM++. It gives
# me the ability to import functions and call them with 
# whatever v0 and vesc parameters I want without needing to change the .py 
# script.
###############################################################################

##########################################
#If you use this code, please cite:

#SHM++: A Refinement of the Standard Halo Model for
#Dark Matter Searches in Light of the Gaia Sausage 
#Evans, O'Hare, McCabe
#arXiv:1810.11468
#DOI:10.1103/PhysRevD.99.023012

#The Earth's velocity for direct detection experiments 
#McCabe
#arXiv:1312.1355
#DOI:10.1088/1475-7516/2014/02/027 
##########################################

##########################################
# Important physical constants
##########################################

'''
#Round Halo parameters
v0R = 220.
vescR = 544.

#SHM++ Halo parameters
v0 = 233.
vesc = 528.
'''
beta = 0.9
eta = 0.2

'''
print("Halo parameters:")
print("beta is", beta)
print("eta is", eta)
'''

#Day=0 is midnight 1 Jan 2020
#Day=60.8 gives the average value of vE
Day = 60.8

'''
print("Days after 1 Jan 2020:", Day)
print()
'''

##########################################
# Parameters for the Monte Carlo integration
##########################################

#number of integration points in Monte Carlo
#higher = more accurate but slower
N = 1000000

#Importance sampling technique
#weight function is exp(-(x**2+y**2+z**2))
# draw random numbers from Normal distribution
mu, sigma = 0, 1/np.sqrt(2) 
x1 = np.random.normal(mu, sigma, N)
y1 = np.random.normal(mu, sigma, N)
z1 = np.random.normal(mu, sigma, N)

def vE(Day,v0):
    ##########################################
    # Calculate the Earths speed
    # See arXiv:1312.1355 for derivation
    ##########################################
    
    uSX = 11.1
    uSY = v0 + 12.24
    uSZ = 7.25

    n = Day + 7303.5 #Day=0 is midnight 1 Jan 2020
    T = (n)/36525.
    e1 = 0.9574 * pi/180.
    L = (280.460 + 0.9856474 * n) * pi/180.
    g = (357.528 + 0.9856003 * n) * pi/180.
    varpi = (282.932 + 0.0000471 * n) * pi/180.
    lam0 = varpi - 270.*pi/180.
    l = L + 2.*e1*sin(g) +5./4.*e1*e1*sin(2.*g)
    UEave = 29.79 #units are km/s

    bX = (5.536+0.013*T) * pi/180.
    bY = (-59.574+0.002*T) * pi/180.
    bZ = (-29.811+0.001*T) * pi/180.

    lamX = (266.840+1.397*T) * pi/180.
    lamY = (347.340+1.375*T) * pi/180.
    lamZ = (180.023+1.404*T) * pi/180.
    
    uEX = UEave * cos(bX) * ( sin(l-lamX) - e1 * cos(lamX-lam0) )
    uEY = UEave * cos(bY) * ( sin(l-lamY) - e1 * cos(lamY-lam0) )
    uEZ = UEave * cos(bZ) * ( sin(l-lamZ) - e1 * cos(lamZ-lam0) )

    vEx = uSX+uEX
    vEy = uSY+uEY
    vEz = uSZ+uEZ
    
    #units are km/s
    return [vEx,vEy,vEz]

def Norm1R_f(vdisR, vescR):
    ##########################################
    # Calculate the fR(v) normalisation factor
    ##########################################

    #weight function is exp(-(x**2+y**2+z**2))
    #introduced dimensionless velocities x y z
    # x = vr / (sqrt2*sigr) etc
    #weight function divides the velocity distribution
    #integral performed in the galactic frame

    vspeed2R = ( (sqrt(2)*vdisR[0]*x1)**2.
                    + (sqrt(2)*vdisR[1]*y1)**2.
                    + (sqrt(2)*vdisR[2]*z1)**2.
                    )
                    
    fintegrandR = 1.
    fconditionR = vescR**2 >= vspeed2R
    sumNR = np.sum(fconditionR.astype(int))

    Norm1R = ( sumNR * pi**(3./2.) * 2.**(3./2.)
                  * vdisR[0] * vdisR[1] * vdisR[2] / N )

    #Exact answer for Norm1R/v0**3 is known:
    #See eg. Appendix B of The Astrophysical Uncertainties Of Dark 
    #Matter Direct Detection Experiments, McCabe, arXiv:1005.0579
    #it is 5.47699 for vesc = 528, v0=233
    #print("Check numerical: round halo normalisation", Norm1R/v0**3.)

    return Norm1R

def Norm1S_f(vdisSaus, vesc):
    ##########################################
    # Calculate the fS(v) normalisation factor
    #########################################

    #same method, weight function etc as Norm1R

    vspeed2S = ( (sqrt(2)*vdisSaus[0]*x1)**2.
                    + (sqrt(2)*vdisSaus[1]*y1)**2.
                    + (sqrt(2)*vdisSaus[2]*z1)**2.
                    )
                    
    fintegrandS = 1.
    fconditionS = vesc**2 >= vspeed2S
    sumNS = np.sum(fconditionS.astype(int))

    Norm1S = ( sumNS * pi**(3./2.) * 2.**(3./2.)
                  * vdisSaus[0] * vdisSaus[1] * vdisSaus[2] / N )

    #Exact answer for Norm1S/v0**3 is known.
    #See eg. Eq. 7 of SHM++: A Refinement of the Standard Halo Model...
    ##Evans, O'Hare, McCabe, arXiv:1810.11468
    #it is 2.09443 for vesc = 528, v0=233, beta=0.9
    #print("Check numerical: Sausage halo normalisation", Norm1S/v0**3.)

    return Norm1S

def gen_round(v0R, vescR, save=False, verbose=True):
    ##########################################
    # Round component
    # Calculate the g(vmin) integral for fR(v)
    # output table of vmin [km/s] and g(vmin) [s/km]
    #########################################

    #weight function is exp(-(x2+y2+z2))
    #integral performed in the galactic frame

    if verbose:
        print("\nCalculating gvmin for Round component")
        #print("vmin [km/s],","g(vmin) [s/km]")

    Lvx = np.linspace(0, v0R*4.6, 300)

    #Round component velocity dispersion 
    sigR = v0R/sqrt(2)
    vdisR = [sigR, sigR, sigR]

    ve = vE(Day,v0R)

    vspeed2R = ( (sqrt(2)*vdisR[0]*x1)**2.
                    + (sqrt(2)*vdisR[1]*y1)**2.
                    + (sqrt(2)*vdisR[2]*z1)**2.
                    )
        
    vspeed2bR = ( (sqrt(2)*vdisR[0]*x1-ve[0])**2.
                     + (sqrt(2)*vdisR[1]*y1-ve[1])**2.
                     + (sqrt(2)*vdisR[2]*z1-ve[2])**2.
                     )
        
    fintegrandR = 1. / np.sqrt( vspeed2bR )

    fcondition1R = vescR**2 >= vspeed2R
    fresult1R = fcondition1R.astype(int)

    Norm1R = Norm1R_f(vdisR, vescR)

    LgvminR = []
    for ivmin in Lvx:
        sat_all_cond = 0.
        vmin = ivmin * 1.
        #do the integral
        fcondition2R = vspeed2bR>=vmin**2
        fresult2R = fcondition2R.astype(int)
        sat_all_cond = fresult1R*fresult2R
        sumNR = np.sum(fintegrandR*sat_all_cond)
        
        int1R = ( sumNR * pi**(3./2.) * 2.**(3./2.)
                 * vdisR[0] * vdisR[1] * vdisR[2] / N )

        gvminR = int1R / (Norm1R)

        LgvminR.append(gvminR)

        #if ivmin % 100 == 0:
        #    print(ivmin,gvminR)

    if save:
        print("Export to gvmin_round_reconcile.dat \n")
        LoutR = list(zip(Lvx,LgvminR))
        savetxt("gvmin_round_reconcile.dat",LoutR)

    return Lvx, LgvminR 

def gen_saus(v0, vesc, save=False, verbose=True):
    ##########################################
    # Sausage component
    # Calculate the g(vmin) integral for fS(v)
    # output table of vmin [km/s] and g(vmin) [s/km]
    #########################################

    #weight function is exp(-(x2+y2+z2))
    #integral performed in the galactic frame

    if verbose:
        print("\nCalculating gvmin for Sausage component")
        #print("vmin [km/s],","g(vmin) [s/km]")

    Lvx = np.linspace(0, v0*4.6, 300)

    #Sausage component velocity dispersion 
    sigrSaus = sqrt( 3. / (2.*(3.-2.*beta)) ) * v0
    sigtSaus = sqrt( 3.*(1.-beta) / (2.*(3.-2.*beta)) ) * v0
    vdisSaus = [sigrSaus, sigtSaus, sigtSaus]

    ve = vE(Day,v0)

    vspeed2S = ( (sqrt(2)*vdisSaus[0]*x1)**2.
                    + (sqrt(2)*vdisSaus[1]*y1)**2.
                    + (sqrt(2)*vdisSaus[2]*z1)**2.
                    )
        
    vspeed2bS = ( (sqrt(2)*vdisSaus[0]*x1-ve[0])**2.
                     + (sqrt(2)*vdisSaus[1]*y1-ve[1])**2.
                     + (sqrt(2)*vdisSaus[2]*z1-ve[2])**2.
                     )
        
    fintegrandS = 1. / np.sqrt( vspeed2bS )

    fcondition1S = vesc**2 >= vspeed2S
    fresult1S = fcondition1S.astype(int)

    Norm1S = Norm1S_f(vdisSaus, vesc) 

    LgvminS = []
    for ivmin in Lvx:
        sat_all_cond = 0.
        vmin = ivmin * 1.
        #do the integral
        fcondition2S = vspeed2bS>=vmin**2
        fresult2S = fcondition2S.astype(int)
        sat_all_cond = fresult1S*fresult2S
        sumNS = np.sum(fintegrandS*sat_all_cond)
        
        int1S = ( sumNS * pi**(3./2.) * 2.**(3./2.)
                 * vdisSaus[0] * vdisSaus[1] * vdisSaus[2] / N )

        gvminS = int1S / (Norm1S)

        LgvminS.append(gvminS)

        #if ivmin % 100 == 0:
        #    print(ivmin,gvminS)

    LgvminS = np.array(LgvminS)

    if save:
        print("Export to gvmin_sausage_reconcile.dat \n")
        LoutS = list(zip(Lvx,LgvminS))
        savetxt("gvmin_sausage_reconcile.dat",LoutS)

    return Lvx, LgvminS

def gen_shmpp(v0, vesc, save=False, plot=False, verbose=True):
    ##########################################
    # Combine round and Sausage components for SHM++
    # output table of vmin [km/s] and g(vmin) [s/km]
    #########################################

    if verbose:
        print("Calculating gvmin for SHM++ ( with eta=",eta,")")

    Lvx = np.linspace(0, v0*4.6, 300)

    _, LgvminR = gen_round(v0, vesc, save, verbose)
    _, LgvminS = gen_saus(v0, vesc, save, verbose)

    LgvminSHMpp = (1.-eta)*np.asarray(LgvminR) + eta*np.asarray(LgvminS)

    if save:
        LoutSHMpp = list(zip(Lvx,LgvminSHMpp))
        print("Export to gvmin_SHMpp_reconcile.dat")
        savetxt("gvmin_SHMpp_reconcile.dat",LoutSHMpp)

    if plot:
        plot(Lvx, LgvminR,"r--", label='Round component')
        plot(Lvx, LgvminS,"b--",  label='Sausage component')
        plot(Lvx, LgvminSHMpp,"k", label='SHM++ with eta='+str(eta))
        legend(loc='upper right')
        xlabel("vmin [km/s]")
        ylabel("g(vmin) [s/km]")
        show()

    return Lvx, LgvminSHMpp
