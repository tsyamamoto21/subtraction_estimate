#!/usr/bin/env python
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from astropy.cosmology import Planck18 as cosmo
import subtraction.population as pop

MAXIMUM_REDSHIFT = 10
MINIMUM_FREQUENCY = 1e-4
MAXIMUM_FREQUENCY = 1e+4
N_FREQUENCY = 200
OBSERVATIONAL_PERIOD = 3


def main(args):

    if args.kind == 'BBH':
        tdmin = 50  # Myr
    elif args.kind == 'BNS':
        tdmin = 20  # Myr
    else:
        raise ValueError('--kind should be BBH or BNS')

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # zmax = config.getfloat('parameters', 'zmax')

    # Calculate formation rate
    zarray_f = np.linspace(0, 20, 1000, endpoint=True)
    formation_rate_density = pop.get_formationratedensity_Vangioni(zarray_f, cosmo)
    formation_rate_density_func = interp1d(zarray_f, formation_rate_density)

    # local_merger_rate_density = config.getfloat('parameters', 'local_merger_rate_density')  # Gpc-3 yr-1
    zarray = np.linspace(0, MAXIMUM_REDSHIFT, 500, endpoint=True)
    normalized_merger_rate_density = pop.get_normalized_merger_rate_density(zarray, tdmin, cosmo, formation_rate_density_func)
    # merger_rate_density = local_merger_rate_density * normalized_merger_rate_density

    with open(f'{outdir}/normalized_merger_rate_density_{args.kind}.dat', 'w+') as fo:
        for i, z in enumerate(zarray):
            fo.write(f'{z}    {normalized_merger_rate_density[i]}\n')

    plt.figure()
    plt.plot(zarray_f, formation_rate_density, c='k', lw=4)
    plt.xlabel('Redshift')
    plt.ylabel('Formation rate density' + r'[$\mathrm{M_\odot\ yr^{-1}\ Mpb^{-3}}$]')
    plt.xlim([0, 20])
    plt.ylim([0, 0.15])
    plt.grid()
    plt.savefig(f'{outdir}/formation_rate_density_{args.kind}.pdf')

    # plt.figure()
    # plt.plot(zarray, merger_rate_density, c='k', lw=4)
    # plt.xlabel('Redshift')
    # plt.ylabel('Merger rate density [Gpc-3 yr-1]')
    # plt.xlim([zarray.min(), zarray.max()])
    # plt.ylim([0, local_merger_rate_density * 2.5])
    # plt.grid()
    # plt.savefig(f'{outdir}/merger_rate_density_{args.kind}.pdf')

    # plt.show()


if __name__ == '__main__':
    import argparse
    # import configparser
    parser = argparse.ArgumentParser(description="Calculate overlap function")
    parser.add_argument('--outdir', type=str, help='Output directory.')
    parser.add_argument('--kind', type=str, help='BBH or BNS')
    args = parser.parse_args()

    # # Generate a ConfigParser object
    # config = configparser.ConfigParser()
    # # Read an ini file
    # if args.kind == 'BBH':
    #     config.read('bbh.ini')
    # elif args.kind == 'BNS':
    #     config.read('bns.ini')
    main(args)
