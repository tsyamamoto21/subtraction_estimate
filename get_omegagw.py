#!/usr/bin/env python
import os
import pickle
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy.cosmology import Planck18 as cosmo
import subtraction.foreground as fg
import subtraction.population as pop
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = (8, 6)

REDSHIFT_RESOLUTION = 1e-3
MINIMUM_REDSHIFT = 0
MAXIMUM_REDSHIFT = 10
MINIMUM_FREQUENCY = 1e-4
MAXIMUM_FREQUENCY = 1e+4
N_FREQUENCY = 200
OBSERVATIONAL_PERIOD = 3
CHANNELS = 8  # (A, E) x 4 triangles
CORRELATIONS = 2  # AA' and EE' from Davide's star


def get_initial_guess_for_zth(snr_threshold, mc, zlim, nchannel=1):
    x0 = ((20.0 / snr_threshold) * (mc / 1.4)**(5.0 / 6.0) / np.sqrt(nchannel))**(15.0 / 13.0)
    if x0 > zlim:
        x0 *= 0.8
    return x0


def main(args):

    snr_threshold = args.snrthreshold

    # PSD
    psd = fg.Sn_DECIGO_SetoYagi

    # Parameters
    local_merger_rate_density = args.local_merger_rate_density

    # Get the merger rate density as a function of a redshift
    normalized_merger_rate_density_file = args.merger_rate_file
    merger_rate_density_norm = np.genfromtxt(normalized_merger_rate_density_file)
    merger_rate_density = merger_rate_density_norm[:, 1] * local_merger_rate_density
    z = merger_rate_density_norm[:, 0]
    # Interpolate
    merger_rate_density_func = interp1d(z, merger_rate_density)

    nsample = args.nsample
    if args.kind == 'bbh':
        massdistribution = pop.GWTC4_BrokenPowerlawPlusTwoPeak()
    elif args.kind == 'bns':
        massdistribution = pop.GWTC4_SimpleUniformBNS()
    else:
        raise ValueError(f'Something wrong with args.kind, {args.kind}')
    samples = massdistribution.get_samples(nsample)
    m1sample = samples[0]
    m2sample = samples[1]

    # Get zbar
    with open(os.path.join(args.outdir, 'overlap_function.pkl'), 'rb') as fo:
        ofdata = pickle.load(fo)
    fsample = ofdata['f']
    zsample = ofdata['z']

    Omega_gw_full = 0
    Omega_gw_unresolvable = 0
    Omega_gw_subthreshold = 0
    Omega_gw_separable = 0
    Omega_gw_err_nonprojected = 0
    Omega_gw_err_projected = 0
    for i in tqdm(range(nsample)):
        # Calculate zupper and zlower
        m1 = m1sample[i]
        m2 = m2sample[i]
        mc = ((m1 * m2)**(3.0 / 5.0)) / ((m1 + m2)**(1.0 / 5.0))
        fmax_insp = (c**3 / (6.0 * np.sqrt(6.0)) / (G * (m1 + m2) * M_sun)).to('s-1').value  # Frequency at LSO
        d2 = 5 * c**5 / (256.0 * np.pi**(8.0 / 3.0) * (G * mc * M_sun)**(5.0 / 3.0))  # \delta_2 in Rosado
        Tmax = cosmo.age(0.0)
        fmin_insp = ((Tmax / d2)**(-3.0 / 8.0)).to('s-1').value
        zupper = fg.get_zupper(fsample, MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT, fmax_insp)
        zlower = fg.get_zlower(fsample, MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT, fmin_insp)

        # Get zbar
        zbarlist = []
        # flgs_all_less_unity = []
        for j in range(N_FREQUENCY):
            n_overlap = ofdata['overlap function'][:, j]
            if np.all(n_overlap < 1.0):
                z_root = zupper[j]
                # flgs_all_less_unity.append(1)
            else:
                n_overlap_interp = interp1d(zsample, n_overlap, bounds_error=False, fill_value='extrapolate')
                z_root = root(lambda z: n_overlap_interp(z) - 1, x0=1.0).x[0]
                # flgs_all_less_unity.append(0)
            zbarlist.append(z_root)

        # Get zbar_threshold
        snr = fg.get_inspiral_snr(fsample, m1, m2, zsample, cosmo, psd, nchannel=CHANNELS)
        if np.all(snr > snr_threshold):
            z_threshold = MAXIMUM_REDSHIFT
        else:
            x0 = get_initial_guess_for_zth(snr_threshold, mc, zlim=MAXIMUM_REDSHIFT, nchannel=CHANNELS)
            z_threshold = root(lambda zi: interp1d(zsample, snr, bounds_error=False, fill_value="extrapolate")(zi) - snr_threshold, x0=x0).x
        zbar_th = np.minimum(z_threshold, zbarlist)

        # Calculate Omega_gw [full, unresolvable]
        Omega_gw_full += fg.get_Omega_gw_with_frequncy_dependent_zrange(fsample, mc, merger_rate_density_func, cosmo, zlower, zupper, dz=REDSHIFT_RESOLUTION)
        Omega_gw_unresolvable += fg.get_Omega_gw_with_frequncy_dependent_zrange(fsample, mc, merger_rate_density_func, cosmo, zbarlist, zupper, dz=REDSHIFT_RESOLUTION)
        Omega_gw_separable += fg.get_Omega_gw_with_frequncy_dependent_zrange(fsample, mc, merger_rate_density_func, cosmo, zmin=zlower, zmax=zbar_th, dz=REDSHIFT_RESOLUTION)
        Omega_gw_subthreshold += fg.get_Omega_gw_with_frequncy_dependent_zrange(fsample, mc, merger_rate_density_func, cosmo, zmin=zbar_th, zmax=zbarlist, dz=REDSHIFT_RESOLUTION)
        Omega_gw_err_nonprojected += fg.get_Omega_error_with_frequncy_dependent_zrange(fsample, m1, m2, merger_rate_density_func, cosmo, zmin=zlower, zmax=zbar_th, psd=psd, nchannel=CHANNELS, dz=REDSHIFT_RESOLUTION, projection=False)
        Omega_gw_err_projected += fg.get_Omega_error_with_frequncy_dependent_zrange(fsample, m1, m2, merger_rate_density_func, cosmo, zmin=zlower, zmax=zbar_th, psd=psd, nchannel=CHANNELS, dz=REDSHIFT_RESOLUTION, projection=True)

    Omega_gw_full /= nsample
    Omega_gw_unresolvable /= nsample
    Omega_gw_subthreshold /= nsample
    Omega_gw_separable /= nsample
    Omega_gw_err_nonprojected /= nsample
    Omega_gw_err_projected /= nsample

    with open(os.path.join(args.outdir, f'omegagw_th{snr_threshold:.1f}_rate{local_merger_rate_density:.1f}_channels{CHANNELS}.txt'), 'w') as fo:
        fo.write('# Frequency    Unresolvable    Separable    Subthreshold    Err    Err(projected)    Full\n')
        for i in range(len(fsample)):
            fo.write(f'{fsample[i]} {Omega_gw_unresolvable[i]} {Omega_gw_separable[i]} {Omega_gw_subthreshold[i]} {Omega_gw_err_nonprojected[i]} {Omega_gw_err_projected[i]} {Omega_gw_full[i]}\n')

    plt.figure()
    plt.plot(fsample, zbar_th, c='k', lw=5, label=r'$\bar{z}_\mathrm{th}$')
    plt.plot(fsample, np.ones_like(fsample) * z_threshold, linestyle='--', label=r'$z_\mathrm{th}$', lw=3)
    plt.plot(fsample, zbarlist, label=r'$\bar{z}$', linestyle=':', lw=3)
    plt.xscale('log')
    plt.xlim([MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY])
    plt.ylim([MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT + 0.5])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Redshift')
    plt.grid()
    plt.grid(which='minor', linestyle=':')
    plt.legend()

    plt.figure()
    plt.loglog(fsample, Omega_gw_full, label='full', lw=6, c='k')
    plt.loglog(fsample, Omega_gw_unresolvable, label='unresolvable', lw=3, linestyle='-.')
    plt.loglog(fsample, Omega_gw_separable, label='separable', lw=3, linestyle='--')
    plt.loglog(fsample, Omega_gw_subthreshold, label='subthreshold', lw=3, linestyle=':')
    plt.loglog(fsample, Omega_gw_err_nonprojected, label='err', lw=3, linestyle='-.')
    plt.loglog(fsample, Omega_gw_err_projected, label='err(projected)', lw=3, linestyle='-.')
    plt.xlim([MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY * 1.5])
    plt.ylim([1.0e-18, 1.0e-9])
    plt.xlabel('Frequency [Hz]')
    plt.ylabel(r'$\Omega_\mathrm{gw}(f)$')
    plt.title(f'SNR threshold = {snr_threshold:.1f}, ' + r'$R_0 = $' + f'{local_merger_rate_density:.1f}' + r'$[\mathrm{Gpc^{-3} yr^{-1}}]$', fontsize=12)
    plt.grid()
    plt.legend(fontsize=12)
    plt.savefig(os.path.join(args.outdir, f'omegagw_th{snr_threshold:.1f}_rate{local_merger_rate_density:.1f}_channels{CHANNELS}.pdf'))
    plt.show()


if __name__ == '__main__':
    import argparse
    # import configparser
    parser = argparse.ArgumentParser(description="Calculate Omega GW")
    parser.add_argument('--outdir', type=str, help='Output directory.')
    parser.add_argument('--kind', type=str, choices=['bbh', 'bns'], help='CBC types, (bbh or bns)')
    parser.add_argument('--merger_rate_file', type=str, help='Path to the file of normalized merger rate density')
    parser.add_argument('--local_merger_rate_density', type=int, help='Local merger rate density in Gpc-3 yr-1')
    parser.add_argument('--nsample', type=int, help='The number of samples for MC integration.')
    parser.add_argument('--snrthreshold', type=int, help='SNR threshold')
    # parser.add_argument('--ndet', type=int, help='The number of baselines')
    args = parser.parse_args()
    main(args)
