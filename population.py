import numpy as np
from scipy.special import erf
from scipy.special import beta as beta_function
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.optimize import root


class BinaryMassDistribution():
    def __init__(self, config, kind):
        self.kind = kind
        if self.kind == 'BBH':
            self.mmin = config.getfloat('population', 'mmin')
            self.mmax = config.getfloat('population', 'mmax')
            self.deltamin = config.getfloat('population', 'deltamin')
            self.alpha = config.getfloat('population', 'alpha')
            self.lmd_peak = config.getfloat('population', 'lmd_peak')
            self.mu_m = config.getfloat('population', 'mu_m')
            self.sigma_m = config.getfloat('population', 'sigma_m')
            self.beta_q = config.getfloat('population', 'beta_q')
            self.qmin = self.mmin / self.mmax

        elif self.kind == 'BNS':
            self.mmin = config.getfloat('population', 'mmin')
            self.mmax = config.getfloat('population', 'mmax')

    def get_samples(self, nsample):
        if self.kind == 'BBH':
            # Get mass distribution functions
            m1grid = np.linspace(self.mmin, self.mmax, 200, endpoint=True)
            qgrid = np.linspace(self.qmin, 1.0, 200, endpoint=True)
            pmass = interp1d(m1grid, get_prob_primary_mass(m1grid, self.mmin, self.mmax, self.deltamin, self.alpha, self.lmd_peak, self.mu_m, self.sigma_m), kind='cubic', bounds_error=False, fill_value='extrapolate')
            m1sample = reject_sampling(pmass, self.mmin, self.mmax, nsample)
            qsample = np.ones(nsample, dtype=np.float64)
            for i in range(nsample):
                pq_conditioned_m1 = interp1d(qgrid, get_prob_massratio_m1conditioned(qgrid, m1sample[i], self.mmin, self.deltamin, self.beta_q))
                qsample[i] = reject_sampling(pq_conditioned_m1, self.qmin, 1.0, 1)
            m2sample = qsample * m1sample

        elif self.kind == 'BNS':
            m1sample = np.random.uniform(self.mmin, self.mmax, (nsample,))
            m2sample = np.random.uniform(self.mmin, self.mmax, (nsample,))
            mask = m1sample < m2sample
            m1_tmp = m1sample.copy()
            m1sample[mask] = m2sample[mask]
            m2sample[mask] = m1_tmp[mask]

        else:
            m1sample = None
            m2sample = None

        return m1sample, m2sample


def reject_sampling(fnc, xmin: float, xmax: float, N: int, umax=None):
    if umax is None:
        x = np.linspace(xmin, xmax, 1000, endpoint=True)
        umax = np.max(fnc(x))
    samples = np.zeros((N,))
    for n in range(N):
        while (True):
            xtrial = np.random.uniform(xmin, xmax)
            utrial = np.random.uniform(0.0, umax)
            p = fnc(xtrial)
            if p > utrial:
                break
        samples[n] = xtrial
    return samples


def get_standard_normal(x):
    return np.exp(-x**2 / 2) / np.sqrt(2.0 * np.pi)


def get_standard_gaussian_cumulative(x):
    return 0.5 * (1.0 + erf(x))


def get_truncated_gaussian(x, a, b, mu, sigma):
    numerator = get_standard_normal((x - mu) / sigma) / sigma
    denominator = get_standard_gaussian_cumulative((b - mu) / sigma) - get_standard_gaussian_cumulative((a - mu) / sigma)
    return numerator / denominator


def get_truncated_powerlaw(x, n, xmin, xmax):
    normalization = (xmax**n - xmin**n) / (n + 1)
    return x**n / normalization


def _get_function_for_cutoff(m, mmin, deltamin):
    # Definition is slightly different
    # S = 1 / (exp(x) + 1) = exp(-x) / (1 + exp(-x))
    eps = 1.0e-5
    return np.exp(- deltamin / (m - mmin + eps) - deltamin / (m - mmin - deltamin + eps))


def get_smooth_cutoff_for_low_mass(m, mmin, deltamin):
    prob = 0
    if isinstance(m, float):
        if m <= mmin:
            prob = 0.0
        elif (mmin < m) * (m < mmin + deltamin):
            x1 = deltamin / (m - mmin)
            x2 = deltamin / (m - mmin - deltamin)
            prob = np.exp(-x1) / (np.exp(-x1) + np.exp(x2))
        else:
            prob = 1.0
    else:
        prob = np.zeros_like(m)
        k = (mmin < m) * (m < mmin + deltamin)
        x1 = deltamin / (m[k] - mmin)
        x2 = deltamin / (m[k] - mmin - deltamin)
        prob[k] = np.exp(-x1) / (np.exp(-x1) + np.exp(x2))
        prob[m >= mmin + deltamin] = 1.0
    return prob


def get_prob_spin_magnitude(x, mean, var):
    alpha = mean * (mean**2 * (1.0 / mean - 1.0) / var - 1.0)
    beta = alpha * (1.0 / mean - 1.0)
    return x**(alpha - 1.0) * (1.0 - x)**(beta - 1.0) / beta_function(alpha, beta)


def get_prob_spin_direction(z1, z2, zeta, sigma_t):
    fieldbinary1 = get_truncated_gaussian(z1, -1, 1, 1.0, sigma_t)
    fieldbinary2 = get_truncated_gaussian(z2, -1, 1, 1.0, sigma_t)
    dynamical1 = 0.5
    dynamical2 = 0.5
    return zeta * fieldbinary1 * fieldbinary2 + (1 - zeta) * dynamical1 * dynamical2


def get_prob_primary_mass(m1, mmin, mmax, deltamin, alpha, lmd_peak, mu_m, sigma_m):
    m1sample = np.arange(mmin, mmax + 0.1, 0.1)
    unnormalized_prob = ((1 - lmd_peak) * get_truncated_powerlaw(m1sample, -alpha, mmin, mmax) + lmd_peak * get_truncated_gaussian(m1sample, mmin, mmax, mu_m, sigma_m)) * get_smooth_cutoff_for_low_mass(m1sample, mmin, deltamin)
    normalization = quad(interp1d(m1sample, unnormalized_prob), m1sample[0], m1sample[-1], limit=100)
    return ((1 - lmd_peak) * get_truncated_powerlaw(m1, -alpha, mmin, mmax) + lmd_peak * get_truncated_gaussian(m1, mmin, mmax, mu_m, sigma_m)) * get_smooth_cutoff_for_low_mass(m1, mmin, deltamin) / normalization[0]


def get_prob_massratio_m1conditioned(q, m1, mmin, deltamin, beta_q):
    qsample = np.linspace(mmin / m1, 0.99, 10000)
    m2 = m1 * qsample
    unnormalized_prob = qsample**beta_q * get_smooth_cutoff_for_low_mass(m2, mmin, deltamin)
    normalization = quad(interp1d(qsample, unnormalized_prob, bounds_error=False, fill_value=0.0), qsample[0], qsample[-1])[0]
    # normalization = np.trapz(unnormalized_prob, qsample)
    return q**beta_q * get_smooth_cutoff_for_low_mass(m1 * q, mmin, deltamin) / (normalization + 1.0e-10)


def get_sfrdensity_MadauDickinson(z, norm=1.0):
    '''
    return: Star formation rate in the unit of Msun yr-1 Mpc-3
    '''
    return norm * (1 + z)**2.7 / (1 + ((1 + z) / 2.9)**5.6)


def get_average_log10metallicity(zarray, cosmo, dz=1e-3):
    metalyield = 0.019
    return_fraction = 0.27
    rhob = (cosmo.Ob0 * cosmo.critical_density0).to('Msun Mpc-3').value
    # Integral
    zmin = np.min(zarray)
    zdummy = np.arange(zmin, 20 + 2 * dz, dz)
    nz = len(zarray)
    integrand = np.tile((get_sfrdensity_MadauDickinson(zdummy, norm=0.015) / (cosmo.H(zdummy).to('yr-1').value) / (1 + zdummy)), (nz, 1))
    mask = np.empty_like(integrand)
    for i in range(nz):
        mask[i] = zdummy >= zarray[i]
    integrand *= mask
    integ = np.trapz(integrand, zdummy, axis=-1)
    return 0.5 + np.log10(metalyield * (1.0 - return_fraction) / rhob * integ)


def get_low_metallicity_fraction(zarray, cosmo, dz=1e-3):
    solar_metallicity = 0.02
    avg_logZ = get_average_log10metallicity(zarray, cosmo, dz)
    return 0.5 * (1.0 + erf(np.sqrt(2.0) * (np.log10(solar_metallicity / 2.0) - avg_logZ)))


def get_formationratedensity_Vangioni(zarray, cosmo, dz=1e-3):
    '''
    BBH formation rate density in the unit of Msun yr-1 Mpc-3
    '''
    # LVK's fiducial model (Callister et al., 1604.02513)
    nu = 0.145
    zm = 1.86
    a = 2.8
    b = 2.62
    Fz = get_low_metallicity_fraction(zarray, cosmo, dz)
    return Fz * nu * a * np.exp(b * (zarray - zm)) / (a - b + b * np.exp(a * (zarray - zm)))


def get_time_delay(z, zf, cosmo):
    '''
    time delay in the unit of Myr
    '''
    td = cosmo.lookback_time(zf) - cosmo.lookback_time(z)
    return td.to('Myr').value


def get_normalized_merger_rate_density(zarray, tdmin, cosmo, formation_rate_density_func, dz=1e-3):
    '''
    tdmin shoud be in the unit of Myr
    '''
    from tqdm import tqdm
    merger_rate_density_integral = np.zeros_like(zarray)
    for i, zsample in enumerate(tqdm(zarray)):
        zf_min = root(lambda zf: get_time_delay(zsample, zf, cosmo) - tdmin, x0=zsample + 0.5).x[0]
        zfsamples = np.arange(zf_min, 20, dz)
        rhofdot = formation_rate_density_func(zfsamples)
        td = get_time_delay(zsample, zfsamples, cosmo)
        integrand = rhofdot / (1 + zfsamples)**2 / cosmo.H(zfsamples).to('Myr-1').value / td
        merger_rate_density_integral[i] = np.trapz(integrand, zfsamples)

    merger_rate_density_integral /= merger_rate_density_integral[0]
    return merger_rate_density_integral
