#!/usr/bin/env python
import os
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
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

    fsample = np.logspace(np.log10(MINIMUM_FREQUENCY), np.log10(MAXIMUM_FREQUENCY), N_FREQUENCY, endpoint=True)

    Omega_gw_full = 0
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

        # Calculate Omega_gw [full, unresolvable]
        Omega_gw_full += fg.get_Omega_gw_with_frequncy_dependent_zrange(fsample, mc, merger_rate_density_func, cosmo, zlower, zupper, dz=REDSHIFT_RESOLUTION)

    # Normalize
    Omega_gw_full /= nsample

    with open(os.path.join(args.outdir, f'fullomegagw_rate{local_merger_rate_density:.1f}.txt'), 'w') as fo:
        fo.write('# Frequency    Full\n')
        for i in range(len(fsample)):
            fo.write(f'{fsample[i]}  {Omega_gw_full[i]}\n')


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
