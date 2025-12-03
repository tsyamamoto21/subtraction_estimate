#!/usr/bin/env python
import os
import pickle
import logging
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy.cosmology import Planck18 as cosmo
from subtraction.foreground import get_overlap_function, get_zlower
import subtraction.population as pop
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = (8, 6)

MINIMUM_REDSHIFT = 0
MAXIMUM_REDSHIFT = 10
REDSHIFT_RESOLUTION = 0.01
MINIMUM_FREQUENCY = 1e-4
MAXIMUM_FREQUENCY = 1e+4
N_FREQUENCY = 200
OBSERVATIONAL_PERIOD = 3


def read_parameters_as_float(section):
    params = {}
    for k, v in section.items():
        params[k] = float(v)
    return params


class FileExistsError(Exception):
    pass


def main(args):
    # Files and directories
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    file_overlap_function = os.path.join(outdir, 'overlap_function.pkl')
    figure_overlap_function = os.path.join(outdir, 'overlap_function.pdf')
    if os.path.exists(file_overlap_function) and (not args.force):
        # logging.error(f"File '{file_overlap_function}' already exists.")
        raise FileExistsError(f"The file '{file_overlap_function}' already exists.")
    if os.path.exists(figure_overlap_function) and (not args.force):
        # logging.error(f"File '{file_overlap_function}' already exists.")
        raise FileExistsError(f"The file '{figure_overlap_function}' already exists.")

    if args.kind == 'bbh':
        massdistribution = pop.GWTC4_BrokenPowerlawPlusTwoPeak()
    elif args.kind == 'bns':
        massdistribution = pop.GWTC4_SimpleUniformBNS()
    else:
        raise ValueError(f'Something wrong with args.kind, {args.kind}')
    samples = massdistribution.get_samples(args.nsample)
    m1sample = samples[0]
    m2sample = samples[1]

    # Get the merger rate density as a function of a redshift
    merger_rate_density_norm = np.genfromtxt(args.merger_rate_file)
    merger_rate_density = merger_rate_density_norm[:, 1] * args.local_merger_rate_density
    z = merger_rate_density_norm[:, 0]
    # Interpolate
    merger_rate_density_func = interp1d(z, merger_rate_density)

    # Calculate overlap function
    Tmax = cosmo.age(0.0)
    zsample = np.arange(MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT + REDSHIFT_RESOLUTION, REDSHIFT_RESOLUTION)
    nz = len(zsample)
    fsample = np.logspace(np.log10(MINIMUM_FREQUENCY), np.log10(MAXIMUM_FREQUENCY), N_FREQUENCY, endpoint=True)
    df = (1.0 / (OBSERVATIONAL_PERIOD * 31557600))
    overlap_function = np.zeros((nz, N_FREQUENCY))
    for i in tqdm(range(args.nsample)):
        m1 = m1sample[i]
        m2 = m2sample[i]
        mc = ((m1 * m2)**(3.0 / 5.0)) / ((m1 + m2)**(1.0 / 5.0))
        d2 = (5 * c**5 / (256.0 * np.pi**(8.0 / 3.0) * (G * mc * M_sun)**(5.0 / 3.0))).decompose().value  # \delta_2 in Rosado
        # fmax_lso = (c**3 / (6.0 * np.sqrt(6.0)) / (G * (m1 + m2) * M_sun)).decompose().value  # Frequency at LSO
        fmin_tmax = ((Tmax / d2)**(-3.0 / 8.0)).decompose().value
        zlower = get_zlower(fsample, MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT, fmin_tmax)
        overlap_function += get_overlap_function(fsample, df, zlower, [MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT], merger_rate_density_func, cosmo, d2, dz=REDSHIFT_RESOLUTION).T
    overlap_function /= args.nsample

    # Save overlap function
    with open(file_overlap_function, 'wb') as fo:
        pickle.dump({'f': fsample, 'z': zsample, 'overlap function': overlap_function}, fo)

    # plot overlap function
    plt.figure()
    plt.pcolormesh(fsample, zsample, np.log10(overlap_function), cmap='seismic', vmin=-3, vmax=3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Redshift')
    plt.xscale('log')
    plt.xlim([MINIMUM_FREQUENCY, MAXIMUM_FREQUENCY])
    plt.ylim([MINIMUM_REDSHIFT, MAXIMUM_REDSHIFT])
    plt.colorbar(label=r'$\log_{10}$ [overlap function]')
    plt.savefig(figure_overlap_function)

    logging.info('Completed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate overlap function")
    parser.add_argument('--kind', type=str, choices=['bns', 'bbh'], help='bns or bbh')
    parser.add_argument('--outdir', type=str, help='Output directory.')
    parser.add_argument('--merger_rate_file', type=str, help='Path to the file of normalized merger rate density')
    parser.add_argument('--local_merger_rate_density', type=int, help='Local merger rate density in Gpc-3 yr-1')
    parser.add_argument('--nsample', type=int, help='The number of samples for MC integration.')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s | %(asctime)s | %(message)s',
                        level=log_level, datefmt='%Y-%m-%d %H:%M:%S')

    main(args)
