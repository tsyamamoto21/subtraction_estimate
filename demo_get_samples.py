#!/usr/bin/env python
import os
# import pickle
import numpy as np
from scipy.interpolate import interp1d
# from scipy.optimize import root
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
import astropy.units as u
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


def get_inspiral_snr(freq, m1, m2, z, cosmo, psd_func, nchannel=1, tobs=3):
    mc = (m1 * m2) ** (3.0 / 5.0) / ((m1 + m2)**(1.0 / 5.0))
    d2 = 5 * c**5 / (256.0 * np.pi**(8.0 / 3.0) * (G * mc * M_sun)**(5.0 / 3.0))  # \delta_2 in Rosado
    fmax = (c**3 / (6.0 * np.sqrt(6.0)) / (G * (m1 + m2) * M_sun)).to('s-1').value  # Frequency at LSO
    fmin = ((tobs * u.yr / d2)**(-3.0 / 8.0)).to('s-1').value
    mask = (fmin / (1 + z) <= freq) * (freq <= fmax / (1 + z))
    integrated_quantity = np.trapz(mask * freq**(-7.0 / 3.0) / psd_func(freq), freq)
    dl = cosmo.luminosity_distance(z)
    factor = ((c / dl)**2 * (G * M_sun * mc * (1 + z) / c**3)**(5 / 3) / (10.0 * np.pi**(4.0 / 3.0))).si.value
    return np.sqrt(nchannel * factor * integrated_quantity)


def main(args):

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
    zsample = pop.reject_sampling(merger_rate_density_func, [MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT], nsample, umax=local_merger_rate_density * 10)

    fsample = np.logspace(np.log10(MINIMUM_FREQUENCY), np.log10(MAXIMUM_FREQUENCY), 10000, endpoint=True)
    snrsample = []
    for i in tqdm(range(nsample)):
        # Calculate zupper and zlower
        m1 = m1sample[i]
        m2 = m2sample[i]
        z = zsample[i]
        # Get zbar_threshold
        snr = get_inspiral_snr(fsample, m1, m2, z, cosmo, psd, nchannel=CHANNELS)
        snrsample.append(snr)

    with open(os.path.join(args.outdir, f'samples_{args.kind}_rate{local_merger_rate_density:.1f}_channels{CHANNELS}.dat'), 'w+') as fo:
        for i in range(nsample):
            fo.write(f'{m1sample[i]}  {m2sample[i]}  {zsample[i]}  {snrsample[i]}\n')

    import math
    bin_min = math.floor(np.log10(np.min(snrsample)))
    bin_max = math.ceil(np.log10(np.max(snrsample)))
    bins = np.logspace(bin_min, bin_max, 50)

    print(np.min(snrsample), np.max(snrsample))
    plt.figure()
    plt.hist(snrsample, bins=bins, histtype='step', color='k', lw=4)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('SNR')
    plt.ylabel('Number of events')
    plt.savefig(os.path.join(args.outdir, f'snrhist_{args.kind}_rate{local_merger_rate_density:.1f}_channels{CHANNELS}.pdf'))
    plt.show()

    plt.figure()
    plt.hist(zsample, bins=30, density=True)
    plt.plot(merger_rate_density_norm[:, 0], merger_rate_density / np.trapz(merger_rate_density, merger_rate_density_norm[:, 0]))
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
    args = parser.parse_args()
    main(args)
