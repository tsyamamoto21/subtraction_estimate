#!/usr/bin/env python
import os
import logging
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

    if args.kind == 'bbh':
        tdmin = 50  # Myr
    elif args.kind == 'bns':
        tdmin = 20  # Myr
    else:
        raise ValueError('--kind should be BBH or BNS')

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    # zmax = config.getfloat('parameters', 'zmax')

    zarray = np.linspace(0, MAXIMUM_REDSHIFT, 500, endpoint=True)
    if args.o4:
        if args.kind == 'bns':
            alpha = 2.6
            beta = 3.6
            zp = 2.2
            # Calculate formation rate
            zarray_f = np.linspace(0, 20, 1000, endpoint=True)
            formation_rate_density = pop.get_normalized_merger_rate_density_lvko4(zarray_f, alpha, beta, zp)
            formation_rate_density_func = interp1d(zarray_f, formation_rate_density)
            normalized_merger_rate_density = pop.get_normalized_merger_rate_density(zarray, tdmin, cosmo, formation_rate_density_func)
        elif args.kind == 'bbh':
            alpha = 3.2
            beta = 5.1
            zp = 2.6
            normalized_merger_rate_density = pop.get_normalized_merger_rate_density_lvko4(zarray, alpha, beta, zp)
        else:
            raise ValueError(f'Something wrong with args.kind, given {args.kind}')

    else:
        # Calculate formation rate
        zarray_f = np.linspace(0, 20, 1000, endpoint=True)
        formation_rate_density = pop.get_formationratedensity_Vangioni(zarray_f, cosmo)
        formation_rate_density_func = interp1d(zarray_f, formation_rate_density)
        normalized_merger_rate_density = pop.get_normalized_merger_rate_density(zarray, tdmin, cosmo, formation_rate_density_func)

    with open(os.path.join(outdir, f'normalized_merger_rate_density_{args.kind}.dat'), 'w+') as fo:
        for i, z in enumerate(zarray):
            fo.write(f'{z}    {normalized_merger_rate_density[i]}\n')

    plt.figure()
    # plt.plot(zarray_f, formation_rate_density, c='k', lw=4)
    plt.plot(zarray, normalized_merger_rate_density, c='k', lw=4)
    plt.xlabel('Redshift')
    plt.ylabel('Normalized merger rate density ' + r'$\mathrm{[Gpc^{-3}\ yr^{-1}]}$]')
    plt.xlim([0, 10])
    plt.ylim([0, np.max(normalized_merger_rate_density) * 1.1])
    plt.grid()
    plt.savefig(os.path.join(outdir, f'normalized_merger_rate_density_{args.kind}.pdf'))

    logging.info('Completed.')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Calculate the merger rate density")
    parser.add_argument('--outdir', type=str, help='Output directory.')
    parser.add_argument('--kind', choices=['bns', 'bbh'], type=str, help='bbh or bns')
    parser.add_argument('--o4', action='store_true', help='Use R(z) of O4a LVK isotropic SGWB paper')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s | %(asctime)s | %(message)s',
                        level=log_level, datefmt='%Y-%m-%d %H:%M:%S')

    main(args)
