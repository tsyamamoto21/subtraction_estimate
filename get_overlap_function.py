#!/usr/bin/env python
import os
import pickle
# import logging
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from astropy.constants import G, c, M_sun
from astropy.cosmology import Planck18 as cosmo
import astropy.units as u
from foreground import get_overlap_function, get_zlower
import population as pop
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['figure.figsize'] = (8, 6)


def read_parameters_as_float(section):
    params = {}
    for k, v in section.items():
        params[k] = float(v)
    return params


class FileExistsError(Exception):
    pass


def main(args, config):

    # # logging
    # logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s', filename='./get_overlap_function.log')

    # Files and directories
    normalized_merger_rate_density_file = config.get('filename', 'normalized_merger_rate_density_file')
    outdir = args.outdir
    file_overlap_function = f'{outdir}/overlap_function.pkl'
    figure_overlap_function = f'{outdir}/overlap_function.pdf'
    if os.path.exists(file_overlap_function) and (not args.force):
        # logging.error(f"File '{file_overlap_function}' already exists.")
        raise FileExistsError(f"The file '{file_overlap_function}' already exists.")
    if os.path.exists(figure_overlap_function) and (not args.force):
        # logging.error(f"File '{file_overlap_function}' already exists.")
        raise FileExistsError(f"The file '{figure_overlap_function}' already exists.")

    # Parameters
    fmin = config.getfloat('parameters', 'fmin')
    fmax = config.getfloat('parameters', 'fmax')
    nf = config.getint('parameters', 'nf')
    tobs = config.getfloat('parameters', 'tobs') * u.yr
    zmin = config.getfloat('parameters', 'zmin')
    zmax = config.getfloat('parameters', 'zmax')
    dz = config.getfloat('parameters', 'dz')
    local_merger_rate_desnity = config.getfloat('parameters', 'local_merger_rate_density')
    nsample = config.getint('parameters', 'nsample')

    massdistribution = pop.BinaryMassDistribution(config, kind=args.kind)
    samples = massdistribution.get_samples(nsample)
    m1sample = samples[0]
    m2sample = samples[1]

    # Get the merger rate density as a function of a redshift
    merger_rate_density_norm = np.genfromtxt(normalized_merger_rate_density_file)
    merger_rate_density = merger_rate_density_norm[:, 1] * local_merger_rate_desnity
    z = merger_rate_density_norm[:, 0]
    # Interpolate
    merger_rate_density_func = interp1d(z, merger_rate_density)

    # Calculate overlap function
    Tmax = cosmo.age(0.0)
    zsample = np.arange(zmin, zmax + dz, dz)
    nz = len(zsample)
    fsample = np.logspace(np.log10(fmin), np.log10(fmax), nf, endpoint=True)
    df = (1.0 / tobs).to('s-1').value
    overlap_function = np.zeros((nz, nf))
    for i in tqdm(range(nsample)):
        m1 = m1sample[i]
        m2 = m2sample[i]
        mc = ((m1 * m2)**(3.0 / 5.0)) / ((m1 + m2)**(1.0 / 5.0))
        d2 = (5 * c**5 / (256.0 * np.pi**(8.0 / 3.0) * (G * mc * M_sun)**(5.0 / 3.0))).decompose().value  # \delta_2 in Rosado
        # fmax_lso = (c**3 / (6.0 * np.sqrt(6.0)) / (G * (m1 + m2) * M_sun)).decompose().value  # Frequency at LSO
        fmin_tmax = ((Tmax / d2)**(-3.0 / 8.0)).decompose().value
        zlower = get_zlower(fsample, zmin, zmax, fmin_tmax)
        overlap_function += get_overlap_function(fsample, df, zlower, [zmin, zmax], merger_rate_density_func, cosmo, d2, dz=dz).T
    overlap_function /= nsample

    # Save overlap function
    with open(file_overlap_function, 'wb') as fo:
        pickle.dump({'f': fsample, 'z': zsample, 'overlap function': overlap_function}, fo)

    # plot overlap function
    plt.figure()
    plt.pcolormesh(fsample, zsample, np.log10(overlap_function), cmap='seismic', vmin=-3, vmax=3)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Redshift')
    plt.xscale('log')
    plt.xlim([fmin, fmax])
    plt.ylim([zmin, zmax])
    plt.colorbar(label=r'$\log_{10}$ [overlap function]')
    plt.savefig(figure_overlap_function)


if __name__ == '__main__':
    import argparse
    import configparser
    parser = argparse.ArgumentParser(description="Calculate overlap function")
    parser.add_argument('--outdir', type=str, help='Output directory.')
    parser.add_argument('--kind', type=str, help='BBH or BNS')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    # Generate a ConfigParser object
    config = configparser.ConfigParser()
    # Read an ini file
    if args.kind == 'BBH':
        config.read('bbh.ini')
    elif args.kind == 'BNS':
        config.read('bns.ini')
    main(args, config)
