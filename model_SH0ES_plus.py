###################################################
# imports

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
from astropy.io import fits

include_MCP = True
include_Miras = True
include_TRGBs = True

###################################################
# prior info

q0 = -0.55
j0 = 1.0
c = 299792.458

mu_N4258 = 29.398
mu_LMC = 18.477
Delta_mu_LMC_SMC = 0.500

sig_N4258 = 0.032
sig_LMC = 0.0263
sig_LMC_SMC = 0.017

mu_SMC = mu_LMC + Delta_mu_LMC_SMC
sig_SMC = np.sqrt(sig_LMC**2.0 + sig_LMC_SMC**2.0)

###################################################
# load data

Y = fits.open('./data_tables/ally_shoes_ceph_topantheonwt6.0_112221.fits')[0].data
L = fits.open('./data_tables/alll_shoes_ceph_topantheonwt6.0_112221.fits')[0].data
C = fits.open('./data_tables/allc_shoes_ceph_topantheonwt6.0_112221.fits')[0].data

###################################################
# parameter dictionary

# Note: N0976 and N9391 are potentially swapped
param_dict = {0: 'mu_M101',
              1: 'mu_Mrk1337',
              2: 'mu_N0691',
              3: 'mu_N1015',
              4: 'mu_N0105',
              5: 'mu_N1309',
              6: 'mu_N1365',
              7: 'mu_N1448',
              8: 'mu_N1559',
              9: 'mu_N2442',
              10: 'mu_N2525',
              11: 'mu_N2608',
              12: 'mu_N3021',
              13: 'mu_N3147',
              14: 'mu_N3254',
              15: 'mu_N3370',
              16: 'mu_N3447',
              17: 'mu_N3583',
              18: 'mu_N3972',
              19: 'mu_N3982',
              20: 'mu_N4038',
              21: 'mu_N4424',
              22: 'mu_N4536',
              23: 'mu_N4639',
              24: 'mu_N4680',
              25: 'mu_N5468',
              26: 'mu_N5584',
              27: 'mu_N5643',
              28: 'mu_N5728',
              29: 'mu_N5861',
              30: 'mu_N5917',
              31: 'mu_N7250',
              32: 'mu_N7329',
              33: 'mu_N7541',
              34: 'mu_N7678',
              35: 'mu_N0976',
              36: 'mu_N9391',
              37: 'mu_N4258',
              38: 'M_H1',
              39: 'mu_LMC',
              40: 'mu_M31',
              41: 'b_W',
              42: 'M_B',
              43: 'Z_W',
              44: 'unknown',
              45: 'Delta_zp',
              46: '5logH0'}

###################################################
# unpack the data [R22 Table 3]

# Cepheids in all SN host galaxies: 2150
Y_CephSNHosts = Y[:2150]

# Cepheids in individual SN host galaxies
Y_CephM101 = Y[:259]
Y_CephMrk1337 = Y[259:274]
Y_CephN0691 = Y[274:302]
Y_CephN1015 = Y[302:320]
Y_CephN0105 = Y[320:328]
Y_CephN1309 = Y[328:381]
Y_CephN1365 = Y[381:427]
Y_CephN1448 = Y[427:500]

Y_CephN1559 = Y[500:610]
Y_CephN2442 = Y[610:787]
Y_CephN2525 = Y[787:860]
Y_CephN2608 = Y[860:882]
Y_CephN3021 = Y[882:898]
Y_CephN3147 = Y[898:925]
Y_CephN3254 = Y[925:973]
Y_CephN3370 = Y[973:1046]

Y_CephN3447 = Y[1046:1147]
Y_CephN3583 = Y[1147:1201]
Y_CephN3972 = Y[1201:1253]
Y_CephN3982 = Y[1253:1280]
Y_CephN4038 = Y[1280:1309]
Y_CephN4424 = Y[1309:1318]
Y_CephN4536 = Y[1318:1358]
Y_CephN4639 = Y[1358:1388]

Y_CephN4680 = Y[1388:1399]
Y_CephN5468 = Y[1399:1492]
Y_CephN5584 = Y[1492:1657]
Y_CephN5643 = Y[1657:1908]
Y_CephN5728 = Y[1908:1928]
Y_CephN5861 = Y[1928:1969]
Y_CephN5917 = Y[1969:1983]
Y_CephN7250 = Y[1983:2004]

Y_CephN7329 = Y[2004:2035]
Y_CephN7541 = Y[2035:2068]
Y_CephN7678 = Y[2068:2084]
###
# Note: the ordering of the next two galaxies is not clear; they may need to be swapped
Y_CephN0976 = Y[2084:2117]
Y_CephN9391 = Y[2117:2150]
###

# Cepheids in NGC 4258: 443
Y_CephN4258 = Y[2150:2593]

# Cepheids in M31: 55
Y_CephM31 = Y[2593:2648]

# Cepheids in LMC: 339
Y_CephLMC = Y[2648:2987]

# Cepheids in SMC: 143
Y_CephSMC = Y[2987:3130]

# SNe Ia in Cepheid host galaxies: 77 measurements for 42 hosts
Y_SNeCephHosts = Y[3130:3207]

# external constraints: 8
Y_external = Y[3207:3215]

# SNe Ia in the Hubble flow: 277
Y_SNe = Y[3215:3492]

###################################################
# repack the data, removing external constraints and unknown parameter

Y_repack = np.zeros(3484)
Y_repack[:3207] = np.copy(Y[:3207])
Y_repack[3207:] = np.copy(Y[3215:])

L_repack = np.zeros((46,3484))
L_repack[:44,:3207] = np.copy(L[:44,:3207])
L_repack[:44,3207:] = np.copy(L[:44,3215:])
L_repack[44:,:3207] = np.copy(L[45:,:3207])
L_repack[44:,3207:] = np.copy(L[45:,3215:])

C_repack = np.zeros((3484,3484))
C_repack[:3207,:3207] = np.copy(C[:3207,:3207])
C_repack[:3207,3207:] = np.copy(C[:3207,3215:])
C_repack[3207:,:3207] = np.copy(C[3215:,:3207])
C_repack[3207:,3207:] = np.copy(C[3215:,3215:])

###################################################
# modify data vector to undo addition of NGC 4258 distance modulus

Y_repack[2150:2593] += mu_N4258

###################################################
# modify data vector to undo addition of LMC distance modulus

Y_repack[2648:2987] += mu_LMC

###################################################
# modify data vector to undo addition of SMC distance modulus

Y_repack[2987:3130] += mu_LMC

###################################################
# notational swap

A = np.copy(L_repack)
C = np.copy(C_repack)
y = np.copy(Y_repack)

###################################################
# add MCP data

if include_MCP:

    # ordering is: UGC 3789, NGC 6264, NGC 5765b, CGCG 074-064, NGC 6323
    mu_MCP = np.array([33.565,35.613,35.250,34.713,35.198])
    sig_MCP = np.array([0.180,0.316,0.102,0.189,0.565])
    z_MCP = np.array([0.0110739,0.0340624,0.028478,0.0239237,0.026086])

    # convert to 5*log(H0)
    y_MCP = (5.0*np.log10(c*z_MCP*(1.0 + ((z_MCP/2.0)*(1.0-q0)) - ((z_MCP**2.0)/6.0)*(1.0 - q0 - (3.0*(q0**2.0)) + j0)))) + 25.0 - mu_MCP

    # add to data vector
    y_new = np.concatenate((y,y_MCP))

    # update design matrix; no new parameters
    A_MCP = np.zeros((A.shape[0],5))
    A_MCP[45,:] = 1.0
    A_new = np.hstack((A,A_MCP))

    # update covariance matrix
    C_MCP = np.diag(sig_MCP**2.0)
    zeromat = np.zeros((5,len(C)))
    C_new = np.block([[C,zeromat.T],[zeromat,C_MCP]])

    # regain notation
    A = np.copy(A_new)
    C = np.copy(C_new)
    y = np.copy(y_new)

###################################################
# add Miras

if include_Miras:

    ############################
    # Miras from Huang+ (2024) -- M101

    # read data table
    P_Miras, m_Miras, sigm_Miras = np.loadtxt('./data_tables/Miras_H18.dat',usecols=(1,4,5),skiprows=1,unpack=True)

    # measurements for fitting
    y_Miras = m_Miras

    # add to data vector
    y_new = np.concatenate((y,y_Miras))

    # update design matrix, including the addition of new parameters for Mira PLR zeropoint and coefficient
    A_Miras = np.zeros((A.shape[0]+2,len(y_Miras)))
    A_Miras[0,:] = 1.0                                # these measurements are only for M101
    A_Miras[-2,:] = 1.0                               # the new PLR zeropoint parameter
    A_Miras[-1,:] = (np.log10(P_Miras) - 2.3)         # the new PLR coefficient parameter
    A_new = np.zeros((A.shape[0]+2,A.shape[1]+len(y_Miras)))
    A_new[:A.shape[0],:A.shape[1]] = np.copy(A)
    A_new[:,A.shape[1]:] = A_Miras

    # update covariance matrix
    C_Miras = np.diag(sigm_Miras**2.0)
    zeromat = np.zeros((len(y_Miras),len(C)))
    C_new = np.block([[C,zeromat.T],[zeromat,C_Miras]])

    # regain notation
    A = np.copy(A_new)
    C = np.copy(C_new)
    y = np.copy(y_new)

    ############################
    # Miras from Huang+ (2020) -- NGC 1559

    # read data table
    P_Miras, m_Miras, sigm_Miras = np.loadtxt('./data_tables/Miras_H20.dat',usecols=(1,4,5),skiprows=1,unpack=True)

    # measurements for fitting
    y_Miras = m_Miras

    # add to data vector
    y_new = np.concatenate((y,y_Miras))

    # update design matrix
    A_Miras = np.zeros((A.shape[0],len(y_Miras)))
    A_Miras[8,:] = 1.0                                # these measurements are only for N1559
    A_Miras[-2,:] = 1.0                               # the new PLR zeropoint parameter
    A_Miras[-1,:] = (np.log10(P_Miras) - 2.3)         # the new PLR coefficient parameter
    A_new = np.zeros((A.shape[0],A.shape[1]+len(y_Miras)))
    A_new[:A.shape[0],:A.shape[1]] = np.copy(A)
    A_new[:,A.shape[1]:] = A_Miras

    # update covariance matrix
    C_Miras = np.diag(sigm_Miras**2.0)
    zeromat = np.zeros((len(y_Miras),len(C)))
    C_new = np.block([[C,zeromat.T],[zeromat,C_Miras]])

    # regain notation
    A = np.copy(A_new)
    C = np.copy(C_new)
    y = np.copy(y_new)

    ############################
    # Miras from Huang+ (2018) -- NGC 4258

    # read data table
    P_Miras, m_Miras, sigm_Miras = np.loadtxt('./data_tables/Miras_H24.dat',usecols=(1,4,5),skiprows=1,unpack=True)

    # measurements for fitting
    y_Miras = m_Miras

    # add to data vector
    y_new = np.concatenate((y,y_Miras))

    # update design matrix
    A_Miras = np.zeros((A.shape[0],len(y_Miras)))
    A_Miras[37,:] = 1.0                                # these measurements are only for N4258
    A_Miras[-2,:] = 1.0                               # the new PLR zeropoint parameter
    A_Miras[-1,:] = (np.log10(P_Miras) - 2.3)         # the new PLR coefficient parameter
    A_new = np.zeros((A.shape[0],A.shape[1]+len(y_Miras)))
    A_new[:A.shape[0],:A.shape[1]] = np.copy(A)
    A_new[:,A.shape[1]:] = A_Miras

    # update covariance matrix
    C_Miras = np.diag(sigm_Miras**2.0)
    zeromat = np.zeros((len(y_Miras),len(C)))
    C_new = np.block([[C,zeromat.T],[zeromat,C_Miras]])

    # regain notation
    A = np.copy(A_new)
    C = np.copy(C_new)
    y = np.copy(y_new)

###################################################
# add TRGBs 

if include_TRGBs:

    ############################
    # TRGBs from Anand+ (2022)

    # ordering is: M101, N1365, N1448, N4038, N4258, N4424, N4536, N5643
    m_TRGB = np.array([25.08, 27.43, 27.35, 27.72, 25.43, 27.05, 27.03, 26.70])
    sigm_TRGB = np.array([0.03, 0.03, 0.04, 0.13, 0.03, 0.05, 0.12, 0.03])
    A_TRGB = np.array([0.013, 0.031, 0.021, 0.071, 0.025, 0.031, 0.028, 0.278])
    color_TRGB = np.array([1.27, 1.28, 1.29, 1.14, 1.32, 1.38, 1.27, 1.30])
    
    # add to data vector
    y_TRGB = m_TRGB - A_TRGB - (0.2*(color_TRGB - 1.23))
    y_new = np.concatenate((y,y_TRGB))

    # update design matrix, including the addition of a new parameter for TRGB absolute magnitude
    A_TRGB = np.zeros((A.shape[0]+1,len(y_TRGB)))
    A_TRGB[0,0] = 1.0           # M101
    A_TRGB[6,1] = 1.0           # N1365
    A_TRGB[7,2] = 1.0           # N1448
    A_TRGB[20,3] = 1.0          # N4038
    A_TRGB[37,4] = 1.0          # N4258
    A_TRGB[21,5] = 1.0          # N4424
    A_TRGB[22,6] = 1.0          # N4536
    A_TRGB[27,7] = 1.0          # N5643
    A_TRGB[-1,:] = 1.0          # the new TRGB absolute magnitude parameter
    A_new = np.zeros((A.shape[0]+1,A.shape[1]+len(y_TRGB)))
    A_new[:A.shape[0],:A.shape[1]] = np.copy(A)
    A_new[:,A.shape[1]:] = A_TRGB

    # update covariance matrix
    C_TRGB = np.diag(sigm_TRGB**2.0)
    zeromat = np.zeros((len(y_TRGB),len(C)))
    C_new = np.block([[C,zeromat.T],[zeromat,C_TRGB]])

    # regain notation
    A = np.copy(A_new)
    C = np.copy(C_new)
    y = np.copy(y_new)

###################################################
# populate the prior info

xp = np.zeros(A.shape[0])
Sinv_diag = np.zeros(A.shape[0])

# distances to anchor galaxies
xp[37] = mu_N4258
xp[39] = mu_LMC
Sinv_diag[37] = 1.0/(sig_N4258**2.0)
Sinv_diag[39] = 1.0/(sig_LMC**2.0)

# Cepheid zeropoint prior from external MW constraints (SH0ES)
xp[38] = -5.903
Sinv_diag[38] = 1.0/(0.024**2.0)
xp[43] = -0.20
Sinv_diag[43] = 1.0/(0.12**2.0)

Sinv = np.diag(Sinv_diag)

###################################################
# compute least squares solution

# Cholesky decomp; more numerically stable than linalg.inv
Cinv = linalg.cho_solve(linalg.cho_factor(C),np.identity(C.shape[0]))

term1 = np.matmul(np.matmul(A,Cinv),A.T)
term2 = term1 + Sinv
term2inv = linalg.cho_solve(linalg.cho_factor(term2),np.identity(term2.shape[0]))
term3 = (np.matmul(np.matmul(A,Cinv),y) + np.matmul(Sinv,xp))

x_sol = np.matmul(term2inv,term3)
covar_sol = np.copy(term2inv)

###################################################
# unpack solution

H0_fit = 10.0**(x_sol[45]/5.0)
H0_err_hi = (10.0**((x_sol[45]+np.sqrt(covar_sol[45,45]))/5.0)) - H0_fit
H0_err_lo = H0_fit - (10.0**((x_sol[45]-np.sqrt(covar_sol[45,45]))/5.0))
H0_err = np.sqrt(H0_err_hi*H0_err_lo)

N4258_fit = x_sol[37]
N4258_err = np.sqrt(covar_sol[37,37])
N4258_fit_Mpc = 10.0**((N4258_fit - 25.0)/5.0)
N4258_err_Mpc_hi = (10.0**(((N4258_fit + N4258_err) - 25.0)/5.0)) - N4258_fit_Mpc
N4258_err_Mpc_lo = N4258_fit_Mpc - (10.0**(((N4258_fit - N4258_err) - 25.0)/5.0))
N4258_err_Mpc = np.sqrt(N4258_err_Mpc_hi*N4258_err_Mpc_lo)

# fake chisq, not accounting for correlated uncertainties
datavec = np.matmul(A.T,x_sol)
chisq = np.sum(((datavec-y)**2.0)/(2.0*np.diag(C)))

print('-'*80)
print('H0: ' + str(H0_fit) + ' +/- ' + str(H0_err) + ' km/s/Mpc')
print('Distance to N4258: ' + str(N4258_fit_Mpc) + ' +/- ' + str(N4258_err_Mpc) + ' Mpc')
print('-'*80)

###################################################
# make residual plot

outname = 'SH0ES'
if include_MCP:
    outname += '+MCP'
if include_Miras:
    outname += '+Miras'
if include_TRGBs:
    outname += '+TRGB'

fig = plt.figure(figsize=(6,3))
ax = fig.add_axes([0.1,0.1,0.6,0.8])
ax2 = fig.add_axes([0.7,0.1,0.2,0.8])

resid_norm = (y - datavec)/np.sqrt(np.diag(C))
hist, hist_edges = np.histogram(resid_norm,bins=40,range=(-4,4),density=True)

ax.plot(resid_norm,'k.',markeredgewidth=0,alpha=0.2)
ax2.plot(hist,0.5*(hist_edges[1:]+hist_edges[:-1]),'k-',drawstyle='steps-mid')

# overplot unit normal
xdum = np.linspace(-4.0,4.0,1000)
ydum = np.exp(-0.5*(xdum**2.0))/np.sqrt(2.0*np.pi)
ax2.plot(ydum,xdum,'r--',linewidth=1)

ax.set_ylabel('Normalized residuals')
ax.set_xlabel('Measurements (mixed types)')

ax.set_ylim(-4,4)
ax2.set_ylim(-4,4)
ax.set_xticks([])
ax2.set_xticks([])
ax2.set_yticklabels([])

xlim = ax.get_xlim()
ax.plot(xlim,[0,0],'k-',linewidth=0.5,zorder=-3)
ax.set_xlim(xlim)

plt.savefig('./plots_residual/residuals_'+outname+'.png',dpi=300,bbox_inches='tight')
plt.close()

###################################################
# cornerplot

contour_vals = [0.5,0.9,0.99]
indlist = [45,37,39,40,38,41,43,42]
paramlabels = [r'$5 \log(H_0)$',r'$\mu_{\rm{N4258}}$',r'$\mu_{\rm{LMC}}$',r'$\mu_{\rm{M31}}$',r'$M_{H,1}^W$',r'$b_W$',r'$Z_W$',r'$M_{B}^{0}$']

if (include_Miras & (not include_TRGBs)):
    contour_vals = [0.5,0.9,0.99]
    indlist = [45,37,39,40,38,41,43,42,46,47]
    paramlabels = [r'$5 \log(H_0)$',r'$\mu_{\rm{N4258}}$',r'$\mu_{\rm{LMC}}$',r'$\mu_{\rm{M31}}$',r'$M_{H,1}^W$',r'$b_W$',r'$Z_W$',r'$M_{B}^{0}$',r'$M_{\rm{Miras}}$',r'$b_{\rm{Miras}}$']

if (include_Miras & include_TRGBs):
    contour_vals = [0.5,0.9,0.99]
    indlist = [45,37,39,40,38,41,43,42,46,47,48]
    paramlabels = [r'$5 \log(H_0)$',r'$\mu_{\rm{N4258}}$',r'$\mu_{\rm{LMC}}$',r'$\mu_{\rm{M31}}$',r'$M_{H,1}^W$',r'$b_W$',r'$Z_W$',r'$M_{B}^{0}$',r'$M_{\rm{Miras}}$',r'$b_{\rm{Miras}}$',r'$M_{\rm{TRGB}}$']

if ((not include_Miras) & include_TRGBs):
    contour_vals = [0.5,0.9,0.99]
    indlist = [45,37,39,40,38,41,43,42,46]
    paramlabels = [r'$5 \log(H_0)$',r'$\mu_{\rm{N4258}}$',r'$\mu_{\rm{LMC}}$',r'$\mu_{\rm{M31}}$',r'$M_{H,1}^W$',r'$b_W$',r'$Z_W$',r'$M_{B}^{0}$',r'$M_{\rm{TRGB}}$']

def initialize_cornerplot(N,figsize=(6,6),xstart=0.05,xend=0.99,xsep=0.01):
    fig = plt.figure(figsize=figsize)
    xwidth = ((xend - xstart) - ((N-1.0)*xsep))/N
    ystart = xstart
    ysep = xsep
    ywidth = xwidth
    axlist2d = list()
    axlist1d = list()
    for ix in range(N):
        axlist_here = list()
        for iy in range(N):
            el1 = xstart + ix*(xsep+xwidth)
            el2 = ystart + iy*(ysep+ywidth)
            el3 = xwidth
            el4 = ywidth
            if (iy < (N-ix-1)):
                axlist_here.append(fig.add_axes([el1,el2,el3,el4]))
            elif (iy == (N-ix-1)):
                axlist1d.append(fig.add_axes([el1,el2,el3,el4]))
        axlist2d.append(axlist_here[::-1])
    return fig, axlist2d, axlist1d

N = len(indlist)
fig, axlist2d, axlist1d = initialize_cornerplot(N,figsize=(12,12),xsep=0.0)

for i in range(N-1):
    for j in range(i+1,N):

        indx = indlist[i]
        indy = indlist[j]
        ax = axlist2d[i][j-i-1]

        sig2x = covar_sol[indx][indx]
        sig2y = covar_sol[indy][indy]
        sigxy = covar_sol[indx][indy]

        lam1 = ((sig2x + sig2y)/2.0) + np.sqrt((((sig2x - sig2y)**2.0)/4.0) + (sigxy**2.0))
        lam2 = ((sig2x + sig2y)/2.0) - np.sqrt((((sig2x - sig2y)**2.0)/4.0) + (sigxy**2.0))
        theta = np.arctan2(lam1-sig2x,sigxy)

        t = np.linspace(0.0,2.0*np.pi,100)

        # plot contours
        for icont, cont_val in enumerate(contour_vals):
            scale = np.sqrt(-2.0*np.log(1.0 - cont_val))
            xel = x_sol[indx] + scale*((np.sqrt(lam1)*np.cos(theta)*np.cos(t)) - (np.sqrt(lam2)*np.sin(theta)*np.sin(t)))
            yel = x_sol[indy] + scale*((np.sqrt(lam1)*np.sin(theta)*np.cos(t)) + (np.sqrt(lam2)*np.cos(theta)*np.sin(t)))
            ax.plot(xel,yel,'k-',linewidth=2.0/(1.5**icont))

        # set axis limits
        xmid = x_sol[indx]
        sigx = np.sqrt(covar_sol[indx][indx])
        xlo = xmid - (4.0*sigx)
        xhi = xmid + (4.0*sigx)
        ymid = x_sol[indy]
        sigy = np.sqrt(covar_sol[indy][indy])
        ylo = ymid - (4.0*sigy)
        yhi = ymid + (4.0*sigy)
        ax.set_xlim(xlo,xhi)
        ax.set_ylim(ylo,yhi)

        if (j != (N-1)):
            # ax.set_xticks(ticks)
            ax.set_xticklabels([])
        else:
            # ax.set_xticks(ticks)
            # ax.set_xticklabels(ticklabels,rotation=rotation)
            ax.set_xlabel(paramlabels[i])
        if (i != 0):
            # ax.set_yticks(ticks)
            ax.set_yticklabels([])
        else:
            # ax.set_yticks(ticks)
            # ax.set_yticklabels(ticklabels)
            ax.set_ylabel(paramlabels[j])
        ax.tick_params(axis='both',direction='in',right=True,top=True)
        ax.grid(linewidth=0.5,linestyle='--',color='gray',alpha=0.2)

# loop through individual parameters
for i in range(N):

    indx = indlist[i]
    ax = axlist1d[i]

    xmid = x_sol[indx]
    sigx = np.sqrt(covar_sol[indx][indx])
    xlo = xmid - (4.0*sigx)
    xhi = xmid + (4.0*sigx)

    xG = np.linspace(xlo,xhi,100)
    yG = np.exp(-0.5*((xG-xmid)/sigx)**2.0) / np.sqrt(2.0*np.pi*(sigx**2.0))

    ax.plot(xG,yG,color='black',linewidth=1)

    # set axis limits
    ax.set_xlim(xlo,xhi)
    if (i != (N-1)):
        # ax.set_xticks(ticks)
        ax.set_xticklabels([])
    else:
        # ax.set_xticks(ticks)
        # ax.set_xticklabels(ticklabels,rotation=rotation)
        ax.set_xlabel(paramlabels[i])
    ax.set_yticks([])
    ax.tick_params(axis='both',direction='in')

    titlestr = paramlabels[i] + '\n' + str(np.round(xmid,3)) + r'$\pm$' + str(np.round(sigx,3))
    ax.set_title(titlestr,fontsize=8)

plt.savefig('./plots_cornerplot/cornerplot_'+outname+'.png',dpi=300,bbox_inches='tight')
plt.close()

