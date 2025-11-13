import numpy as np
from scipy.integrate import cumulative_trapezoid
from astropy.constants import G, c, M_sun
import astropy.units as u


def get_Omega_gw_with_frequncy_dependent_zrange(farray, mc, merger_rate_density_func, cosmo, zmin, zmax, dz=0.001):

    # Check that farray, zmin, zmax has the same shape
    assert len(farray) == len(zmin), f'farray and zmin do not have the same shape, farray has {len(farray)} and zmin has {len(zmin)}'
    assert len(zmin) == len(zmax), f'zmin and zmax do not have the same shape, zmin has {len(zmin)} and zmax has {len(zmax)}'

    Omega_gw = np.zeros_like(farray)
    for j, fsample in enumerate(farray):
        zsample = np.arange(zmin[j], zmax[j], dz)
        Hubble = cosmo.H(zsample)
        hubble_unit = Hubble.unit
        Hubble_value = Hubble.value
        integrand = merger_rate_density_func(zsample) / (1.0 + zsample)**(4.0 / 3.0) / Hubble_value
        integrated_value = np.trapz(integrand, zsample)
        integrated_value = integrated_value / hubble_unit / (u.Gpc**3) / (u.yr)
        factor = (np.pi**(2.0 / 3.0) / 3.0) * (G * M_sun * mc)**(5.0 / 3.0) * (fsample * u.Hz)**(2.0 / 3.0) / G / cosmo.critical_density0 / c**2
        Omega_gw[j] = (factor * integrated_value).decompose()
    return Omega_gw


def get_zlower(farray, zmin, zmax, fmin):
    k1 = farray <= (fmin / (1 + zmax))
    k2 = ((fmin / (1 + zmax)) < farray) * (farray < (fmin / (1 + zmin)))
    k3 = (fmin / (1 + zmin)) <= farray
    zlower = np.zeros_like(farray)
    zlower[k1] = zmax
    zlower[k2] = fmin / (farray[k2]) - 1.0
    zlower[k3] = zmin
    return zlower


def get_zupper(farray, zmin, zmax, fmax):
    k1 = farray <= fmax / (1 + zmax)
    k2 = ((fmax / (1 + zmax)) < farray) * (farray < (fmax / (1 + zmin)))
    k3 = fmax / (1 + zmin) <= farray
    zupper = np.zeros_like(farray)
    zupper[k1] = zmax
    zupper[k2] = fmax / (farray[k2]) - 1.0
    zupper[k3] = zmin
    return zupper


def Sn_DECIGO(freq):
    """
    DECIGO Science paper
    """
    fp = 7.5
    return 1.8e-47 * (1 + (freq / fp)**2.0) + 1.2e-50 * (freq**(-4.)) / (1 + (freq / fp)**2.) + 1.0e-46 * ((freq / 0.1)**(-16.))


def Sn_DECIGO_SetoYagi(freq):
    fp = 7.36
    term1 = 7.05e-48 * (1.0 + (freq / fp)**2)
    term2 = 4.8e-51 * freq**(-4.0) / (1.0 + (freq / fp)**2)
    term3 = 5.33e-52 * freq**(-4.0)
    return term1 + term2 + term3


# def get_bns_snrsq(freq, mc, z, cosmo, psd, ndet=1):
#     integrand = freq**(-7.0 / 3.0) / psd
#     integrated_quantity = np.trapz(integrand, freq) * (u.Hz**(-1.0 / 3.0))
#     dl = cosmo.luminosity_distance(z)
#     factor = (c / dl)**2 * (G * M_sun * mc * (1 + z) / c**3)**(5 / 3) / (10.0 * np.pi**(4.0 / 3.0))
#     return ndet * factor.to('Hz(1/3)') * integrated_quantity


def get_inspiral_snr(freq, m1, m2, z, cosmo, psd_func, nchannel=1, tobs=3):
    nz = len(z)
    mc = (m1 * m2) ** (3.0 / 5.0) / ((m1 + m2)**(1.0 / 5.0))
    d2 = 5 * c**5 / (256.0 * np.pi**(8.0 / 3.0) * (G * mc * M_sun)**(5.0 / 3.0))  # \delta_2 in Rosado
    fmax = (c**3 / (6.0 * np.sqrt(6.0)) / (G * (m1 + m2) * M_sun)).to('s-1').value  # Frequency at LSO
    fmin = ((tobs * u.yr / d2)**(-3.0 / 8.0)).to('s-1').value
    freq = np.tile(freq, (nz, 1))
    mask = (fmin / (1 + z.reshape(nz, 1)) <= freq) * (freq <= fmax / (1 + z.reshape(nz, 1)))
    integrated_quantity = np.trapz(mask * freq**(-7.0 / 3.0) / psd_func(freq), freq, axis=-1)
    dl = cosmo.luminosity_distance(z)
    factor = ((c / dl)**2 * (G * M_sun * mc * (1 + z) / c**3)**(5 / 3) / (10.0 * np.pi**(4.0 / 3.0))).si.value
    return np.sqrt(nchannel * factor * integrated_quantity)


def get_snr_of_sgwb(freq, omega_gw, psd, cosmo, duration=3, orf=1.0, ncorrelation=1):
    factor = (3.0 / 10.0 / np.pi**2) * (cosmo.H0**2) * np.sqrt(2.0 * duration * u.yr)
    integrand = orf**2 * omega_gw**2 / (freq**6) / (psd**2)
    integrated_quantity = np.trapz(integrand, freq) / (u.Hz**3)
    return (factor * np.sqrt(integrated_quantity)).decompose() * np.sqrt(ncorrelation)


def get_overlap_function(freq, df, zlow, zrange, merger_rate_density_func, cosmo, d2, dz=0.001):
    nf = len(zlow)
    zsample = np.arange(zrange[0], zrange[1] + dz, dz)
    nz = len(zsample)
    Qfactor = np.tile((freq**(-8.0 / 3.0) - (freq + df)**(-8.0 / 3.0)), (nz, 1)).T
    # taue = get_time_spent_frequency_interval(freq, df, zsample, d2)
    dVdz = cosmo.differential_comoving_volume(zsample) * (4.0 * np.pi * u.steradian)
    Ndot = (merger_rate_density_func(zsample) * (u.Gpc**(-3)) * (u.yr**(-1)) * dVdz).to('s-1').value
    integrand = np.tile(Ndot / (1 + zsample)**(8.0 / 3.0), (nf, 1))
    for j in range(nf):
        kless = zsample <= zlow[j]
        integrand[j, kless] = 0.0
    overlap_function = cumulative_trapezoid(integrand, zsample, axis=-1, initial=0) * d2 * Qfactor
    return overlap_function


def get_Omega_error_with_frequncy_dependent_zrange(farray, m1, m2, merger_rate_density_func, cosmo, zmin, zmax, psd, Np=11, nchannel=1, tobs=3, dz=0.001, projection=False):

    # Check that farray, zmin, zmax has the same shape
    assert len(farray) == len(zmin), f'farray and zmin do not have the same shape, farray has {len(farray)} and zmin has {len(zmin)}'
    assert len(zmin) == len(zmax), f'zmin and zmax do not have the same shape, zmin has {len(zmin)} and zmax has {len(zmax)}'

    mc = ((m1 * m2)**(3.0 / 5.0)) / ((m1 + m2)**(1.0 / 5.0))
    Omega_gw = np.zeros_like(farray)
    zsample = np.arange(np.min(zmin), np.max(zmax), dz)
    # Calculate the factor of subtraction errors
    snrsqlist = get_inspiral_snr(farray, m1, m2, zsample, cosmo, psd, nchannel, tobs)**2
    if projection:
        residual_factor = (Np / np.array(snrsqlist))**2
    else:
        residual_factor = Np / np.array(snrsqlist)
    # Get integrand
    Hubble = cosmo.H(zsample)
    hubble_unit = Hubble.unit
    Hubble_value = Hubble.value
    integrand = residual_factor * merger_rate_density_func(zsample) / (1.0 + zsample)**(4.0 / 3.0) / Hubble_value

    for j, fsample in enumerate(farray):
        mask = (zmin[j] <= zsample) * (zsample <= zmax[j])
        integrated_value = np.trapz(integrand * mask, zsample)
        integrated_value = integrated_value / hubble_unit / (u.Gpc**3) / (u.yr)
        factor = (np.pi**(2.0 / 3.0) / 3.0) * (G * M_sun * mc)**(5.0 / 3.0) * (fsample * u.Hz)**(2.0 / 3.0) / G / cosmo.critical_density0 / c**2
        Omega_gw[j] = (factor * integrated_value).decompose()
    return Omega_gw
