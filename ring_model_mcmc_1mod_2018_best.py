import numpy as np
import emcee
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
import pdb
import model_azimuth3_1mod as mod
import oitools
import corner
from multiprocessing import Pool
import os
from matplotlib import gridspec
os.environ["OMP_NUM_THREADS"] = "1"

def lnlike (param, u, v, v2=None, v2err=None, cp=None, cperr=None, sta_index_v=None, sta_index_cp=None, \
                  flag_vis2=None, flag_cp=None, wave=np.array(0.1)):
    theta1, incl, c1, s1, la, lkr, fs, fd = param  #Order with POS1,2,...

    sf_u = np.zeros([u.shape[0], wave.shape[0]]) #spatial frequency in u
    sf_v = np.zeros([v.shape[0], wave.shape[0]])
    visamp_tot = np.zeros([v.shape[0], wave.shape[0]])
    cp_mod_tot = np.zeros([cp.shape[0], wave.shape[0]])
    for i in range(len(wave)):
        sf_u[:, i] = np.array(u / wave[i])
        sf_v[:, i] = np.array(v / wave[i])
        vis = mod.model_fourier3(sf_u[:, i], sf_v[:, i], theta1, incl, c1, s1, la, lkr, 1.0, fs, fd, (1-fs-fd)) #Llama al modelo de disco
            #Result vis -> Complex visibility
        visamp_tot[:, i] = np.absolute(vis) #square of total Visibility per wavelength

        visamp = np.absolute(vis)
        visphi = np.angle(vis, deg=True) #Phases angle
        visamp = visamp.reshape(-1, 1)
        visphi = visphi.reshape(-1, 1)
        cp_mod = oitools.compute_closure_phases(6, 4, sta_index_v, sta_index_cp, visamp, visphi) # Calculate the closure phases for a given phases and amplitudes
            #It use the number of baseline, # of CP per snapshot, station of cp, and v.
        cp_mod_tot[:, i] = np.squeeze(cp_mod) #Total CP per wavelength

    [ind_v2] = np.where(flag_vis2.reshape(-1) == False)
    [ind_cp] = np.where(flag_cp.reshape(-1) == False)
    sv2 = v2err.reshape(-1)[ind_v2]*0+0.05**2
    scp = cperr.reshape(-1)[ind_cp]#*0+0.5*4


#------------------------------------ Chi2

    # visamp = np.squeeze(visamp[ind_v2])
    # cp_mod = np.squeeze(cp_mod[ind_cp])
    chi2_v = (v2.reshape(-1)[ind_v2] - (visamp_tot.reshape(-1)[ind_v2]) ** 2) / sv2

    cp_data_temp = cp.reshape(-1)[ind_cp]
    cp_mod_temp = cp_mod_tot.reshape(-1)[ind_cp]

    ind11 = np.where(cp_data_temp < 0)
    ind12 = np.where(cp_mod_temp < 0)

    cp_data_temp[ind11] = cp_data_temp[ind11] + 360
    cp_mod_temp[ind12] = cp_mod_temp[ind12] + 360

    chi2_cp = (((cp_mod_temp - cp_data_temp) + 180) % 360 - 180)
    chi2_cp_final = np.deg2rad(chi2_cp) / np.deg2rad(scp)

    # chi2_v_tot = -0.5 * np.sum(chi2_v**2 + np.log(sv2**2) + np.log(2*np.pi))/ chi2_v.shape[0]
    # chi2_cp_final_tot = -0.5 * np.sum(chi2_cp_final**2 + np.log(np.deg2rad(scp)**2) + np.log(2*np.pi))/chi2_cp_final.shape[0]
    chi2_v_tot = -0.5 * np.sum(chi2_v ** 2) / chi2_v.shape[0]
    chi2_cp_final_tot = -0.5 * np.sum(chi2_cp_final ** 2) / chi2_cp_final.shape[0]

    chi2_tot = chi2_v_tot + chi2_cp_final_tot
    if chi2_v_tot > 0:
        print('------------- a ver el chi2_V', chi2_v_tot, chi2_cp_final_tot)
    if chi2_cp_final_tot > 0:
        print('************* a ver el chi2_cp', chi2_v_tot, chi2_cp_final_tot)
    #print(chi2_tot)
    
    return chi2_tot

# Define the probability function as likelihood * prior.
def lnprior(param): #Function Log of prior distribution -> Insert the range of the parameters
    theta1, incl, c1, s1, la, lkr, fs, fd = param
    if ( 160 < theta1 < 245) and ( 1 < incl < 60 ) and (-3.0 < s1 < -1.1) and (-1.4 < c1 < -0.75)  and (1.0 < la < 1.3) and \
            (-1.2 < lkr < -0.5) and (0 < (fs + fd) <= 1) and (fd > 0) and (fs > 0) and (2.8<(s1**2 + c1**2)<3.05):
   
        return 0.0
    else:
        return -np.inf

def lnprob(param, u, v, vis2, vis2_err, cp, cp_err, sta_index_v,sta_index_cp, flag_vis2, flag_cp, wave):
    lp = lnprior(param)
    if lp != 0:
        return -np.inf
    else:
        return lnlike(param, u, v, vis2, vis2_err, cp, cp_err, sta_index_v, sta_index_cp, flag_vis2, flag_cp, wave)

#Load data

if __name__ == "__main__":

    oi_file = 'RCra/FT_DATA/COMB/2018_205_COMB_RCra.fits'
    year = '2018_205_1mod_chi_'
    observables = oitools.extract_data(oi_file) #Read the file and extract all data in the dir variable 'observables'

    uur = observables['u']
    vvr = observables['v']
    vis2 = observables['vis2']
    vis2_err = observables['vis2_err']
    cp = observables['t3']
    cp_err = observables['t3_err']
    sta_index_v = observables['sta_vis']
    sta_index_cp = observables['sta_cp']
    flag_vis2 = observables['flag_vis2']
    flag_cp = observables['flag_t3']
    uv_cp = observables['uv_cp']
    uv = observables['uv']
    wave = observables['waves']

#Prior parameters uniform distribution EMCEE

    ndim, nwalkers = 8, 200 #Parameters and MC
    nsteps = 6000
    pos1 = np.random.uniform(160,245, size=nwalkers) #theta1 (position angle)
    pos2 = np.random.uniform(1, 60, size=nwalkers)  #incl (inclination)
    pos3 = np.random.uniform(-1.4, -0.75, size=nwalkers) #C1 (cosine of the modulation)
    pos4 = np.random.uniform(-3.0, -1.1, size=nwalkers) #S1 (sine of the modulation)
    pos5 = np.random.uniform(1.0,1.3, size=nwalkers)  # la (log of the disk size)
    pos6 = np.random.uniform(-1.2, -0.5, size=nwalkers)  # lkr (log of the kernel size)
    pos7 = np.random.uniform(0.1, 0.3, size=nwalkers)  # fs (flux of the star)
    pos8 = np.random.uniform(0.5, 1.1, size=nwalkers)  # fd (flux of the disk)
    #pos9 = np.random.uniform(0.6, 1.5, size=nwalkers)  # C2 (cosine of the modulation)
    #pos10 = np.random.uniform(-0.4, 0.4, size=nwalkers)  # S2 (sine of the modulation)
    pos = np.array([pos1, pos2, pos3, pos4, pos5, pos6, pos7, pos8]).T
    print('Shape posssssssssssssssss')
    print(pos.shape)


    with Pool() as pool: #Paraleliza
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=( uur, vvr, vis2, vis2_err, cp, cp_err, sta_index_v, \
                                                                      sta_index_cp, flag_vis2, flag_cp, wave), pool=pool)
        
        sampler.run_mcmc(pos, nsteps, progress=True)
    
    samples = sampler.get_chain(flat=True, thin=1, discard= int(0.7*nsteps))
    probs = sampler.get_log_prob(flat=True, thin=1, discard=int(0.7*nsteps))
    samples = samples[np.where(probs>5*np.max(probs))]


    print(type(samples))
    print(samples.shape)
    print(np.max(probs), ' A ver la probbbbbbbb ------*******')
    #print(np.max(samples) , np.max(probs))

    # 'c2', 's2'
    # c2_mcmc[0], s2_mcmc[0]]
    theta1_mcmc, incl_mcmc, c1_mcmc, s1_mcmc, la_mcmc, lkr_mcmc, fs_mcmc, fd_mcmc = map(lambda v: (v[1], v[2] - v[1], v[1] - v[0]), #, c2_mcmc, s2_mcmc
                                   zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    fig = corner.corner(samples, labels=['theta', 'incl', 'c1', 's1', 'la', 'lkr', 'fs', 'fd'], \
                        show_titles=True, title_fmt="0.2e",
                        levels=(1 - np.exp(-0.5), 1 - np.exp(-2.0)), \
                        truths=[theta1_mcmc[0], incl_mcmc[0], c1_mcmc[0], s1_mcmc[0], la_mcmc[0], \
                                lkr_mcmc[0], fs_mcmc[0], fd_mcmc[0]], quantiles=[0.16, 0.5, 0.84], title_kwargs={
            "fontsize": 12})  # range=[(1.555, 1.575), (0.074, 0.076), (15.9, 15.95), (-25.7, -25.5)]
    fig.savefig("205_test_2018_1mod_chi_.png")

    fig2, axes = plt.subplots(10, figsize=(10, 7), sharex=True)
    labels = ['theta', 'incl', 'c1', 's1', 'la', 'lkr', 'fs', 'fd'] #'c2', 's2'
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    fig2.savefig("205_samples_2018_1mod_chi_.png")


##########
    fig3 = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])
    ax4 = plt.subplot(gs[3])

    for i in range(len(wave)):
        vis = mod.model_fourier3(uur / wave[i], vvr / wave[i], theta1_mcmc[0], incl_mcmc[0], c1_mcmc[0], s1_mcmc[0], \
                                 la_mcmc[0], lkr_mcmc[0], 1.0, \
                                fs_mcmc[0], fd_mcmc[0], 1-fs_mcmc[0]-fd_mcmc[0])
        im = mod.model_im3(512, 1.0 / mod.mas2rad(0.1 * 512), theta1_mcmc[0], incl_mcmc[0], c1_mcmc[0], s1_mcmc[0], \
                                 la_mcmc[0], lkr_mcmc[0], 1.0, \
                                fs_mcmc[0], fd_mcmc[0], 1-fs_mcmc[0]-fd_mcmc[0], wave[i], year ) #, c2_mcmc[0], s2_mcmc[0]

        visamp = np.absolute(vis)
        visphi = np.angle(vis, deg=True)
        visamp = visamp.reshape(-1, 1)
        visphi = visphi.reshape(-1, 1)
        cp_mod = oitools.compute_closure_phases(6, 4, sta_index_v, sta_index_cp, visamp, visphi)
        rr = np.sqrt((uur / wave[i]) ** 2 + (vvr / wave[i]) ** 2)

        if len(wave) == 1:
            [ind_v2] = np.where(flag_vis2 == False)
            [ind_cp] = np.where(flag_cp == False)
            # colorVal = scalarMap.to_rgba(values[i])
            ax1.errorbar(rr[ind_v2], vis2[ind_v2], yerr=vis2_err[ind_v2], fmt='o', color='darkblue')
            ax1.plot(rr[ind_v2], visamp[ind_v2] ** 2, 'o', color='crimson', zorder=200,
                     label=str(np.round(wave[i] / 1e-6, 4)) + ' $\mu$m')
            ax2.errorbar(observables['uv_cp'][ind_cp], cp[ind_cp], yerr=cp_err[ind_cp], fmt='o', color='darkblue')
            ax2.plot(observables['uv_cp'][ind_cp], cp_mod[ind_cp], 'o', color='crimson', zorder=200)
            ax3.plot(rr[ind_v2], (vis2[ind_v2] - np.squeeze(visamp[ind_v2] ** 2)), 'o', color='crimson')
            ax4.plot(observables['uv_cp'][ind_cp], (cp[ind_cp] - np.squeeze(cp_mod[ind_cp])), 'o', color='crimson')
        else:
            [ind_v2] = np.where(flag_vis2[:, i] == False)
            [ind_cp] = np.where(flag_cp[:, i] == False)
            # colorVal = scalarMap.to_rgba(values[i])
            ax1.errorbar(rr[ind_v2], vis2[ind_v2, i], yerr=vis2_err[ind_v2, i], fmt='o', color='darkblue')
            ax1.plot(rr[ind_v2], visamp[ind_v2] ** 2, 'o', color='crimson', zorder=200,
                     label=str(np.round(wave[i] / 1e-6, 4)) + ' $\mu$m')
            ax2.errorbar(observables['uv_cp'][ind_cp, i], cp[ind_cp, i], yerr=cp_err[ind_cp, i], fmt='o', color='darkblue')
            ax2.plot(observables['uv_cp'][ind_cp, i], cp_mod[ind_cp], 'o', color='crimson', zorder=200)
            ax3.plot(rr[ind_v2], (vis2[ind_v2, i] - np.squeeze(visamp[ind_v2] ** 2)), 'o', color='crimson')
            ax4.plot(observables['uv_cp'][ind_cp, i], (cp[ind_cp, i] - np.squeeze(cp_mod[ind_cp])), 'o', color='crimson')
            # ax1.annotate('AZIMUTHAL MODULATED DISK MODEL', xy=(0.2, .90), xycoords='axes fraction', xytext=(0.2, 0.90), \
            #             textcoords='axes fraction', color='black', fontsize=14)

    ax3.set_ylabel('M-D')
    ax4.set_ylabel('M-D')
    ax1.set_ylim([0, 0.5])
    ax2.set_ylim([-100, 100])
    ax3.set_ylim([-0.1, 0.1])
    ax4.set_ylim([-5, 5])
    ax3.set_xlabel('Spatial Freq. [1/rad]')
    ax1.set_ylabel('Squared Visibility')
    ax4.set_xlabel('Spatial Freq. [1/rad]')
    ax2.set_ylabel('Closure Phases [deg]')
    yticks = ax3.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    yticks = ax4.yaxis.get_major_ticks()
    yticks[-1].label1.set_visible(False)
    fig3.legend()
    #fig3.suptitle(tit)
    fig3.subplots_adjust(hspace=0.0)

    fig3.savefig('model_azimuth_2018_205_1mod_chi_.pdf', bbox_inches='tight')
    plt.show()
    pdb.set_trace()
        
    #print(sampler.acor())