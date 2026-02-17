# -*- coding: utf-8 -*-
"""
@author: D.Pfeiffer, D.Derr & L.Lind
"""
import warnings
from scipy.optimize import curve_fit
from scipy.integrate import quad
import numpy as np
import logging

logger = logging.getLogger(__name__)

logger.info("Calling helper_functions.py.")

# route all warnings through the logging module
logging.captureWarnings(True)


##############################################
### linear combiniation related quantities ###
##############################################
def plain_lambdas_to_rel(lambda_plus, lambda_minus):
    return [(lambda_plus+lambda_minus)/2, lambda_plus-lambda_minus]


def rel_lambdas_to_plain(lambda_mean, lambda_diff):
    return [lambda_mean + lambda_diff/2, lambda_mean - lambda_diff/2]


def rel_lambdas_to_amplitude(theta, A0, lambda_mean, lambda_diff):
    return abs(A0 * np.sqrt(4*np.cos(theta/2)**2*lambda_mean**2 + np.sin(theta/2)**2*lambda_diff**2))


def sigma_density(sigma, lambda_plus, lambda_minus):
    """
    Calculate the effective standard deviation of the density function based on the weights lambda_plus and lambda_minus.
    
    Parameters
    ----------
    sigma : float
        Standard deviation of S+ resp. S- signals.
    lambda_plus : float
        Weight for the S+ signal.
    lambda_minus : float
        Weight for the S- signal.

    Returns
    -------
    sigma_eff : float
        Effective standard deviation of the density function.
    """
    return sigma * np.sqrt(lambda_plus**2 + lambda_minus**2)


def amp_max_guess(A0, lambda_plus, lambda_minus):
    """
    Provides an upper bound for the amplitude based on the given lambda parameters and A0.

    Parameters
    ----------
    A0 : float
        Amplitude of indiviudal signals S+ and S-.
    lambda_plus : float
        Lambda parameter for the S+ signal.
    lambda_minus : float
        Lambda parameter for the S- signal.
    
    Returns
    -------
    A_max : float
        Upper bound for the amplitude based.
    """
    A_max_1 = rel_lambdas_to_amplitude(
        0, A0, *plain_lambdas_to_rel(lambda_plus, lambda_minus))
    A_max_2 = rel_lambdas_to_amplitude(
        np.pi, A0, *plain_lambdas_to_rel(lambda_plus, lambda_minus))
    A_max = np.max([A_max_1, A_max_2])
    return A_max


#########################
### signal generation ###
#########################
def generate_signals(A0, sigma, theta, n_phis, lambda_plus, lambda_minus, seed):
    """
    Generate numerical replication if signals S+, S-, their linear combinations S_sum and S_diff.

    Parameters
    ----------
    A0 : float
        Amplitude of the underlying signals S+ and S-.
    sigma : float
        Standard deviation of the Gaussian noise added to S+ and S-.
    theta : float
        Differential phase between S+ and S- in radians.
    n_phis : int
        Number of random laser phases to simulate.
    lambda_plus : float
        Weight for the S+ signal in the linear combinations.
    lambda_minus : float
        Weight for the S- signal in the linear combinations.
    seed : int
        Random seed for reproducibility.
    
    Returns
    -------
    S_plus : np.ndarray
        Simulated S+ signals with shape (n_phis,).
    S_minus : np.ndarray
        Simulated S- signals with shape (n_phis,).
    S_sum : np.ndarray
        Linear combination of S+ and S- defined as lambda_plus * S_plus + lambda_minus * S_minus.
    S_diff : np.ndarray
        Linear combination of S+ and S- defined as lambda_plus * S_plus - lambda_minus * S_minus.
    """
    rng = np.random.default_rng(seed)
    phis = rng.uniform(0, 2 * np.pi, n_phis)
    B_plus = rng.normal(0, sigma, n_phis)
    B_minus = rng.normal(0, sigma, n_phis)

    S_plus = A0 * np.cos(phis + theta/2) + B_plus
    S_minus = A0 * np.cos(phis - theta/2) + B_minus

    S_sum = lambda_plus * S_plus + lambda_minus * S_minus
    S_diff = lambda_plus * S_plus - lambda_minus * S_minus

    return (S_plus, S_minus, S_sum, S_diff)

####################################
### amplitude related functions  ###
####################################
def Amp_all(t, A0, a, p0, dp, tau=100e-6, k=(4*np.pi/780.226e-9)):
    """
    Calculate the time-dependent amplitude of S_all.

    Parameters
    ----------
    t : float
        Interferometer time in seconds.
    A0 : float
        Initial amplitude.
    a : float
        Measured acceleration in m/s^2.
    p0: float
        Measured population in the mf=0 substate
    dp: float
        Measured population imbalance between the mf=+/-1 states.
    tau : float
        Pulse width. The default is 100µs
    k : float, optional
        Wavenumber k in 1/m. The default is 4*np.pi/780.226e-9.

    Returns
    -------
    A : float
        Time-dependent amplitude of S_all.

    """
    theta = 2 * (a*k*t**2 + k*a*t*0.149*tau)
    Ac = A0 * (np.cos(theta/2) + p0 * (1-np.cos(theta/2)))
    As = A0 * dp * np.sin(theta/2)
    
    A = np.sqrt(Ac**2 + As**2)
    
    return A

def Amp_sum(t, A0, a, tau=100e-6, k=(4*np.pi/780.226e-9)):
    """
    Calculate the time-dependent amplitude of the sum signal.

    Parameters
    ----------
    t : float
        Interferometer time in seconds.
    A0 : float
        Initial amplitude.
    a : float
        Measured acceleration in m/s^2.
    tau : float
        Pulse width. The default is 100 µs
    k : float, optional
        Wavenumber k in 1/m. The default is 4*np.pi/780.226e-9.

    Returns
    -------
    A : float
        Time-dependent amplitude of S_sum.

    """
    theta = 2*(a*k*t**2 + k*a*t*0.149*tau)
    A = np.sqrt(2) * A0 * abs(np.cos(theta/2))
    return A

def amplitude_to_theta(A, A0, lambda_mean, lambda_diff):
    """
    Calculate the differential phase theta from the fitted amplitude A.

    Parameters
    ----------
    A : float
        Fitted/measured amplitude.
    A0 : float
        Amplitude of the underlying signals S+ resp. S-.
    lambda_mean : float
        Mean of the lambda parameters used in the linear combinations.
    lambda_diff : float
        Difference of the lambda parameters used in the linear combinations.
    
    Returns
    -------
    theta : float
        Calculated differential phase in radians. Returns np.nan if the input parameters lead to an invalid calculation.
    """
    numerator = A0**2 * (lambda_diff**2 + 4 *
                         lambda_mean**2) - 2 * np.array(A)**2
    denominator = A0**2 * (lambda_diff**2 - 4 * lambda_mean**2)
    quotient = numerator / denominator
    quotient_safe = np.where(np.abs(quotient) <= 1, quotient, np.nan)
    return np.arccos(quotient_safe)

###################################
### histogram related functions ###
###################################
def create_hist(data):
    """
    Create a normalised histogram of the given data with an automatically determined number of bins
    based on the square root of the number of data points.

    Parameters
    ----------
    data : array-like
        Input data for which the histogram is to be created.
    
    Returns
    -------
    bin_centres : np.ndarray
        Centres of the histogram bins.
    hist_vals : np.ndarray
        Normalised histogram values (density).
    bin_edges : np.ndarray
        Edges of the histogram bins.
    """
    n_points = len(data)

    data_min = np.min(data)
    data_max = np.max(data)
    data_centre = (data_min + data_max) / 2
    data_width = np.abs(data_max - data_min)

    n_bins_hist = 2*int(np.ceil(np.sqrt(n_points)))

    hist_vals, bin_edges = np.histogram(
        data,
        bins=n_bins_hist,
        range=(data_centre - data_width, data_centre + data_width),
        density=True
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return (bin_centres, hist_vals, bin_edges)


def create_hist_not_normalised(data):
    """
    Create a histogram of the given data with an automatically determined number of bins based on the square root of the number of data points.

    Parameters
    ----------
    data : array-like
        Input data for which the histogram is to be created.
    
    Returns
    -------
    bin_centres : np.ndarray
        Centres of the histogram bins.
    hist_vals : np.ndarray
        Histogram values (not normalised).
    bin_edges : np.ndarray
        Edges of the histogram bins.
    """
    n_points = len(data)

    data_min = np.min(data)
    data_max = np.max(data)
    data_centre = (data_min + data_max) / 2
    data_width = np.abs(data_max - data_min)

    n_bins_hist = 2*int(np.ceil(np.sqrt(n_points)))

    hist_vals, bin_edges = np.histogram(
        data,
        bins=n_bins_hist,
        range=(data_centre - data_width, data_centre + data_width)
    )

    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return (bin_centres, hist_vals, bin_edges)


def moving_average(x, w):
    """
    Compute the moving average of a 1D array x with a window size w.

    Parameters
    ----------
    x : array-like
        Input 1D array for which the moving average is to be computed.
    w : int
        Window size for the moving average. Must be a positive integer.
    
    Returns
    -------
    moving_avg : np.ndarray
        Array of the same shape as x containing the moving average values.
    """
    return np.convolve(x, np.ones(w)/w, mode='same')


def hist_density_scalar_robust(u, A, sigma, mu):
    """
    Robust version of histogram density calculation with improved numerical integration.
    Handles boundary singularities, narrow Gaussians, adaptive domain focusing.

    Parameters
    ----------
    u : float
        Point at which to evaluate the density.
    A : float
        Amplitude parameter of the density function.
    sigma : float
        Standard deviation of the baseline fluctuations used in the density function.
    mu : float
        Mean of the signal/histogram represented by the density function.
    
    Returns
    -------
    density : float
        Evaluated density at point u.
    """
    precision = np.finfo(float).eps  # machine precision

    ### handle degenerate cases ###
    ## A too small => only Gaussian of width sigma ##
    if A < precision:
        # if Gaussian to narrow => delta(u - mu) #
        if sigma < precision:
            if abs(u - mu) < precision:
                return 1.0
            else:
                return 0.0
        # regular Gaussian #
        else:
            return 1/(np.sqrt(2*np.pi)*sigma) * np.exp(-(u - mu)**2/(2*sigma**2))

    ## sigma initially too small => Gaussian becomes delta(u - z - mu) ##
    elif sigma < precision:
        if abs(u-mu) < A:
            return 1/(np.pi * np.sqrt(A**2 - (u-mu)**2))
        else:
            return 0.0

    ### no degenerate cases ###
    else:
        ### transform integrand ###
        ## substiution z := A sin(t) => dz = A |cos(t)| dt ##
        ## |cos(t)| = cos(t) since np.arcsin in [-pi/2,pi/2]##

        normalisation = 1 / (np.pi*sigma*np.sqrt(2*np.pi))

        def integrand_transformed(t):
            exponent = -((u - mu - A * np.sin(t))**2) / (2 * sigma**2)
            # cos(t) cancels out the denominator
            return np.exp(exponent)

        ### find t_min and t_max depending on fraction of Gaussian wrt to interval [-A, A] ###
        # since if fraction too small no good integration
        # compare FWHM = 2.355 * sigma (contains 98 % of the Gaussian) to 2 * A => sigma / A as measure

        ## full integration range [-A, A] ##
        t_min = -np.pi/2
        t_max = np.pi/2

        ## sigma / A < 1e-2 as threshold, because of default numerical integration in quad ##
        ## guarantees that enough quadraature points are sampled within the significant region of the Gaussian ##
        if sigma / A < 1e-2:
            gaussian_centre = u - mu
            # integrate within +- n_sigma_range sigma of Gaussian centre
            n_sigma_range = 9
            # find new lower bound, which is bounded by -A below
            z_min = max(-A, gaussian_centre - n_sigma_range * sigma)
            # lower bound must also be less than A
            z_min = min(z_min, A)
            # find new upper bound, which is bounded by A above
            z_max = min(A, gaussian_centre + n_sigma_range * sigma)
            # upper bound must also be greater than -A
            z_max = max(z_max, -A)

            if z_max <= z_min:
                return 0.0

            # overwrite t_min and t_max to adjust integration domain #
            t_min = np.arcsin(z_min/A)
            t_max = np.arcsin(z_max/A)

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("always")  # ensure warnings are emitted
                integral_result, _ = quad(integrand_transformed, t_min, t_max,
                                          epsabs=1e-12, epsrel=1e-10, limit=1000)
                return normalisation * integral_result

        except Exception:
            # Logs the full traceback
            logger.exception(
                f"Integration error for u={u}, A={A}, sigma={sigma}, mu={mu}"
            )
            # Return fallback value or re-raise as needed
            return np.nan


def hist_density(u, A, sigma, mu):
    """
    Vectorised version of robust histogram density calculation.
    """
    vec_func = np.vectorize(hist_density_scalar_robust, otypes=[float],
                            signature='(),(),(),()->()')
    return vec_func(u, A, sigma, mu)


def hist_density_for_fitting(u, A, sigma, mu):
    """
    Histogram density for fitting that ensures A and sigma are non-negative by taking their absolute values before calculating the density.

    Parameters
    ----------
    u : array-like
        Points at which to evaluate the density.
    A : float
        Amplitude parameter of the density function (will be treated as non-negative).
    sigma : float
        Standard deviation of the baseline fluctuations used in the density function (will be treated as non-negative).
    mu : float
        Mean of the signal/histogram represented by the density function.
    
    Returns
    -------
    density : np.ndarray
        Evaluated density at points u.
    """
    A = np.abs(A)
    sigma = np.abs(sigma)
    return hist_density(u, A, sigma, mu)


#######################
### fitting routine ###
#######################
def fit_hist_density(x, y, p0=[0.5, 0.05, 0.0], add_info="", A_max=None, bounds=True):
    """
    Fit histogram density with fallback jacobian calculation.

    First attempts fit without explicit jac parameter (uses default numerical jacobian).
    If that fails, retries with jac='3-point' for more robust jacobian calculation.

    Parameters
    ----------
    x : array-like
        Points at which the density is evaluated (e.g. bin centres).
    y : array-like
        Histogram density values at points x.
    p0 : list or array-like, optional
        Initial guess for the parameters [A, sigma, mu]. The default is [0.5, 0.05, 0.0].
    add_info : str, optional
        Additional information to include in log messages for debugging. The default is "".
    A_max : float, optional
        Optional upper bound for the amplitude parameter A. If provided,
        A will be constrained to be less than or equal to A_max. The default is None (no upper bound).
    bounds : bool, optional
        Whether to apply bounds to the parameters during fitting.
        If True, A will be constrained to be non-negative and less than or equal to A_max (if provided),
        and sigma will be constrained to be non-negative. The default is True.

    Returns
    -------
    popt : np.ndarray
        Optimal values for the parameters [A, sigma, mu] found by the fit.
    ssr : float
        Sum of squared residuals for the fit.
    """
    info_text = f": {add_info}" if add_info else ""
    A_upper_limit = np.inf
    if A_max is not None:
        A_upper_limit = A_max

        if p0[0] < 0:
            p0[0] = 0

        if p0[0] > A_max:
            p0[0] = A_max - 1e-12

    # build bounds depending on flag
    if bounds:
        lower_bounds = np.array([0, 0, -np.inf], dtype=float)
        upper_bounds = np.array([A_upper_limit, np.inf, np.inf], dtype=float)
        fit_bounds = (lower_bounds, upper_bounds)
    else:
        fit_bounds = (-np.inf, np.inf)

    # first try without explicit jac parameter
    try:
        popt, _, infodict, _, _ = curve_fit(
            hist_density_for_fitting,
            x, y, p0=p0,
            bounds=fit_bounds,
            full_output=True,
            maxfev=2000)

        fvec = infodict['fvec']
        ssr = np.dot(fvec, fvec)  # sum of squared residuals

    except Exception:
        logger.exception(f"First curve_fit attempt failed{info_text}.")
        # second try with explicit jac parameter jac='3-point'
        try:
            popt, _, infodict, _, _ = curve_fit(
                hist_density_for_fitting,
                x, y, p0=p0,
                bounds=fit_bounds,
                jac='3-point',
                full_output=True,
                maxfev=2000)
            
            fvec = infodict['fvec']
            ssr = np.dot(fvec, fvec)

        except Exception:
            logger.exception(
                f"Second curve_fit attempt with jac='3-point' failed{info_text}.")

            popt = np.full(len(p0), np.nan)
            ssr = np.nan

    return (popt, ssr)


def fit_routine_hist(signal, add_info="", log_infos=False, forced_initial_guess=None, sigma_guess=None, A_max=None, bounds=True):
    """
    Fit the histogram density of the given signal with robust initial parameter estimation and optional bounds.

    Parameters
    ----------
    signal : array-like
        Input signal for which the histogram density is to be fitted.
    add_info : str, optional
        Additional information to include in log messages for debugging. The default is "".
    log_infos : bool, optional
        Whether to log detailed information about the fitting process. The default is False.
    forced_initial_guess : list or array-like, optional
        If provided, this initial guess [A, sigma, mu] will be used for the fit instead of the automatically estimated one.
        The default is None (use automatic estimation).
    sigma_guess : float, optional
        If provided, this value will be used as the initial guess for sigma in the automatic estimation process.
        The default is None (estimate sigma automatically).
    A_max : float, optional
        Optional upper bound for the amplitude parameter A during fitting. If provided,
        A will be constrained to be less than or equal to A_max. The default is None (no upper bound).
    bounds : bool, optional
        Whether to apply bounds to the parameters during fitting.
        If True, A will be constrained to be non-negative and less than or equal to A_max (if provided),
        and sigma will be constrained to be non-negative. The default is True.
    
    Returns
    -------
    bin_centres : np.ndarray
        Centres of the histogram bins.
    hist_vals : np.ndarray
        Normalised histogram values (density) at the bin centres.
    fit_results : np.ndarray
        Optimal values for the parameters [A, sigma, mu] found by the fit.
    peak_position : float
        Position of the peak in the histogram (estimated from the data).
    peak_sigma_position : float
        Position corresponding to one sigma away from the peak in the direction determined by the data (estimated from the data).
    initial_estimates : list
        Initial estimates for the parameters [A, sigma, mu] used for the fit (either automatically estimated or forced).
    ssr : float
        Sum of squared residuals for the fit.
    """
    bin_centres, hist_vals, _ = create_hist(signal)

    # estimate initial parameters for the fit, if not forced by argument
    if forced_initial_guess is None:
        mean_estimate_raw = np.mean(signal)

        smooth_hist_vals = moving_average(hist_vals, w=1)  # window width = 1

        index_of_centre = int(len(hist_vals)/2)

        peak_index = np.argmax(smooth_hist_vals)
        peak_value = smooth_hist_vals[peak_index]
        peak_position = bin_centres[peak_index]

        # determine direction for sigma estimation based on whether peak is to the left or right of centre
        if peak_index >= index_of_centre:
            sign_for_sigma_direction = +1
        else:
            sign_for_sigma_direction = -1

        if sigma_guess is not None:
            sigma_estimate_raw = sigma_guess
        else:
            sigma_est_fraction = 1/4

            if peak_index >= index_of_centre:
                # right peak => search for falling values to the right
                # mask for all indices to the right of the peak where hist_vals <= peak_value * sigma_est_fraction
                mask = (smooth_hist_vals[peak_index:] <=
                        peak_value * sigma_est_fraction)
                # find first True entry
                rel_idx = np.argmax(mask)
                if not mask.any():
                    idx_of_peak_frac = None  # no entries found under peak_value * sigma_est_fraction
                else:
                    idx_of_peak_frac = peak_index + rel_idx
            else:
                # left peak => search for falling values to the left
                mask = (smooth_hist_vals[:peak_index+1] <=
                        peak_value * sigma_est_fraction)[::-1]
                rel_idx = np.argmax(mask)
                if not mask.any():
                    idx_of_peak_frac = None
                else:
                    idx_of_peak_frac = peak_index - rel_idx

            if idx_of_peak_frac is None:
                print("no point of decrease found")
            else:
                peak_frac_position = bin_centres[idx_of_peak_frac]
                distance = abs(peak_frac_position - peak_position)
            sigma_estimate_raw = distance * \
                np.sqrt(-1/(2*np.log(sigma_est_fraction)))

        amplitude_estimate_raw = np.abs(
            peak_position - mean_estimate_raw) + sigma_estimate_raw
        
        peak_sigma_position = peak_position + \
                sign_for_sigma_direction*sigma_estimate_raw

        # amplitude_estimate, sigma_estimate, mean_estimate = round_to_significant_digits_vec(
        #     [amplitude_estimate_raw, sigma_estimate_raw, mean_estimate_raw], digits=3)
        
        amplitude_estimate, sigma_estimate, mean_estimate = amplitude_estimate_raw, sigma_estimate_raw, mean_estimate_raw
    else:
        amplitude_estimate, sigma_estimate, mean_estimate = forced_initial_guess
        peak_position = mean_estimate + amplitude_estimate
        peak_sigma_position = peak_position + sigma_estimate

    if log_infos:
        logger.info(
            f"Estimates: {amplitude_estimate:.17g}, {sigma_estimate:.17g}, {mean_estimate:.17g}")

    fit_results, ssr = fit_hist_density(
        bin_centres,
        hist_vals,
        p0=[amplitude_estimate, sigma_estimate, mean_estimate],
        add_info=add_info,
        A_max=A_max,
        bounds=bounds
    )

    if log_infos:
        logger.info(
            f"Final Fit: {fit_results[0]:.17g}, {fit_results[1]:.17g}, {fit_results[2]:.17g}, SSR = {ssr:.17g}.")

    return (bin_centres, hist_vals, fit_results, peak_position, peak_sigma_position, [amplitude_estimate, sigma_estimate, mean_estimate], ssr)

def round_to_significant_digits_vec(x, digits=3):
    """
    Rounds each element of the input array x to the specified number of significant digits, while preserving the shape of the input. NaN values are preserved as NaN in the output.

    Parameters
    ----------
    x : array-like
        Input array to be rounded.
    digits : int, optional
        Number of significant digits to round to. The default is 3.
    
    Returns
    -------
    rounded_x : np.ndarray
        Array of the same shape as x with each element rounded to the specified number of significant digits.
    """
    x = np.asarray(x)

    def round_elem(val):
        if np.isnan(val):  # check for NaN values
            return np.nan
        if val == 0:
            return 0
        order = np.floor(np.log10(abs(val)))
        decimals = int(digits - order - 1)
        return round(val, decimals)
    # vectorise the scalar function
    vfunc = np.vectorize(round_elem)
    return vfunc(x)

##################################
### ellipse related functions  ###
##################################

class FitEllipseError(Exception):
    """Custom exception for fit_ellipse errors."""
    pass


def fit_ellipse(x, y, add_info=""):
    """
    Fit an ellipse to the given points (x, y) and ensure the result is a real-valued
    numpy array of shape (6,).
    In case of an error, FitEllipseError is raised and an np.array with np.nan is returned.
    Based on fit_ellipse() from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/.

    Parameters
    ----------
    x : array-like
        x-coordinates of the points, length >= 3
    y : array-like
        y-coordinates of the points, length >= 3

    Returns
    -------
    params : np.ndarray, shape (6,)
        Ellipse parameters [A, B, C, D, E, F] or np.nan array in case of error.
    """
    # Helper: returns np.nan-Array (6,)
    def nan_params():
        return np.full((6,), np.nan, dtype=float)

    try:
        # input validation
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1:
            raise FitEllipseError(
                "x and y must be 1-dimensional arrays.")
        if x.size != y.size:
            raise FitEllipseError(
                "x and y must have the same number of elements.")
        if x.size < 3:
            raise FitEllipseError(
                "At least 3 points are required to fit an ellipse.")

        D1 = np.vstack([x**2, x*y, y**2]).T
        D2 = np.vstack([x, y, np.ones_like(x)]).T
        S1 = D1.T @ D1
        S2 = D1.T @ D2
        S3 = D2.T @ D2

        # check for singularity
        if np.linalg.cond(S3) > 1e12:
            raise FitEllipseError("Matrix S3 is almost singular."+add_info)

        T = -np.linalg.inv(S3) @ S2.T
        M = S1 + S2 @ T

        C = np.array([[0, 0, 2],
                      [0, -1, 0],
                      [2, 0, 0]], dtype=float)

        # check for singular C matrix
        if np.linalg.cond(C) > 1e12:
            raise FitEllipseError("Matrix C is almost singular."+add_info)

        M = np.linalg.inv(C) @ M

        eigval, eigvec = np.linalg.eig(M)
        con = 4 * eigvec[0] * eigvec[2] - eigvec[1]**2

        idx = np.where(con > 0)[0]
        if idx.size == 0:
            raise FitEllipseError(
                "No valid eigenvectors found (ellipse not defined)."+add_info)
        ak = eigvec[:, idx]

        raw_params = np.concatenate((ak, T @ ak)).ravel()

        # output validation
        if not isinstance(raw_params, np.ndarray) or raw_params.shape != (6,):
            raise FitEllipseError(
                f"Unexpected output shape: {raw_params.shape}"+add_info)
        if not np.isrealobj(raw_params):
            raise FitEllipseError(
                "Found parameters are not purely real."+add_info)

        return raw_params

    except FitEllipseError as e:
        # Exception werfen und nan-Array zurückliefern
        logger.error(f"FitEllipseError: {e}")
        return nan_params()



def parametric_to_polar_vectorised(coeffs):
    """
    Vectorised version of parametric_to_polar that can handle numpy arrays.

    This function converts ellipse parameters from the general conic form:
    ax² + bxy + cy² + dx + fy + g = 0

    to polar form parameters: centre (x0, y0), semi-axes (ap, bp), and orientation (phi).

    Based on cart_to_pol() from https://scipython.com/blog/direct-linear-least-squares-fitting-of-an-ellipse/.

    Parameters:
    -----------
    coeffs : np.ndarray
        Coefficients array of shape (..., 6) where the last dimension contains
        the 6 ellipse coefficients [a, b, c, d, f, g].

        Can handle any shape as long as the last dimension is 6, including:
        - (6,) for a single ellipse
        - (n_thetas, n_stoch_rep, 6) for multiple ellipses
        - Any other multidimensional array with last dimension = 6

    Returns:
    --------
    x0, y0, ap, bp, phi : tuple of np.ndarray
        - x0, y0: Center coordinates, shape matches input[:-1] 
        - ap: Semi-major axis length (always >= bp)
        - bp: Semi-minor axis length (always <= ap)
        - phi: Orientation angle in radians [0, π)

        All outputs have the same shape as the input coeffs array,
        except the last dimension is removed.

    Raises:
    -------
    ValueError: If any coefficient set does not represent a valid ellipse
    """
    # Validate input
    if coeffs.shape[-1] != 6:
        raise ValueError(
            f"Last dimension must be 6 (got shape {coeffs.shape})")

    # Extract coefficients - note the divisions by 2 match the original function
    a = coeffs[..., 0]
    b = coeffs[..., 1] / 2
    c = coeffs[..., 2]
    d = coeffs[..., 3] / 2
    f = coeffs[..., 4] / 2
    g = coeffs[..., 5]

    # Calculate discriminant - must be negative for ellipse
    den = b**2 - a*c
    invalid_mask = den > 0
    if np.any(invalid_mask):
        raise ValueError(
            f'{np.sum(invalid_mask)} coefficient sets do not represent ellipses: '
            f'b²-ac must be negative (discriminant of conic)!'
        )

    # Calculate ellipse centre
    x0 = (c*d - b*f) / den
    y0 = (a*f - b*d) / den

    # Calculate numerator for semi-axes calculation
    num = 2 * (a*f**2 + c*d**2 + g*b**2 - 2*b*d*f - a*c*g)

    # Calculate factor for eigenvalue-based semi-axes
    fac = np.sqrt((a - c)**2 + 4*b**2)

    # Calculate semi-axis lengths from eigenvalues
    ap_temp = np.sqrt(np.abs(num / den / (fac - a - c)))
    bp_temp = np.sqrt(np.abs(num / den / (-fac - a - c)))

    # Ensure ap is the major axis (larger) and bp is the minor axis (smaller)
    width_gt_height = ap_temp >= bp_temp
    ap = np.where(width_gt_height, ap_temp, bp_temp)
    bp = np.where(width_gt_height, bp_temp, ap_temp)

    # Calculate orientation angle phi
    phi = np.where(
        b == 0,
        np.where(a < c, 0.0, np.pi/2),        # Case when b = 0
        np.arctan((2.0 * b) / (a - c)) / 2    # Case when b ≠ 0
    )

    # Apply corrections to phi based on eigenvalue ordering and axis swapping
    phi = np.where(a > c, phi + np.pi/2, phi)
    phi = np.where(~width_gt_height, phi + np.pi/2, phi)
    phi = phi % np.pi  # Ensure phi is in [0, π)

    return x0, y0, ap, bp, phi


def conic_section_to_theta(A, B, C):
    """
    Calculates the differential phase theta from the conic section parameters A, B, C of an ellipse using the formula:
    theta_ell_recon = arccos(-B / (2 * sqrt(A * C)))

    Parameters
    ----------
    A : array-like
        Coefficient A of the conic section.
    B : array-like
        Coefficient B of the conic section.
    C : array-like
        Coefficient C of the conic section.
    
    Returns
    -------
    theta_ell_recon : np.ndarray
        Estimated differential phase theta in radians.
        Returns np.nan for entries where the calculation is invalid (e.g. A*C <= 0 or arccos argument out of bounds).
    """
    A = np.asarray(A)
    B = np.asarray(B).copy()
    C = np.asarray(C)

    # sign change: everywhere where A < 0, B is reversed
    B = np.where(A < 0, -B, B)

    # initialise array with NaNs
    theta_ell_recon = np.full(A.shape, np.nan, dtype=float)

    # mask for valid values (A*C > 0)
    valid = (A * C) > 0

    # calculate theta_ell_recon only for valid entries
    if np.any(valid):
        numerator = -B[valid]
        denominator = 2 * np.sqrt(A[valid] * C[valid])
        for_arccos = numerator / denominator

        # arccos argument must be in [-1, 1] for real results
        within_bounds = np.abs(for_arccos) <= 1

        theta_ell_recon[valid] = np.nan  # caution: first NaN
        theta_ell_recon[np.where(valid)[0][within_bounds], np.where(
            valid)[1][within_bounds]] = np.arccos(for_arccos[within_bounds])

    return theta_ell_recon

## analytic expression for principal axes ##
def axis_sum(theta, A0, tol=1e-8):
    """
    True axis sum of the ellipse for a given theta and A0.

    Parameters
    ----------
    theta : array-like
        Differential phase in radians.
    A0 : float
        Amplitude of the underlying signals S+ resp. S-.
    tol : float, optional
        Tolerance for handling singularities at theta = 0, pi. The default is 1e-8.
    
    Returns
    -------
    axis_sum : np.ndarray
        The true axis sum of the ellipse for the given theta and A0.
    """
    theta_half = np.asarray(theta)/2
    theta_half_mod = np.mod(theta_half, np.pi)
    d0 = np.minimum(theta_half_mod, np.pi - theta_half_mod)
    d_pi2 = np.abs(theta_half_mod - np.pi/2)
    mask = (d0 < tol) | (d_pi2 < tol)
    result = np.zeros_like(theta_half)
    result[mask] = 0.0
    valid_idx = ~mask
    theta_half_valid = theta_half[valid_idx]
    sin_2theta_half = np.sin(2 * theta_half_valid)
    csc_2theta_half = 1 / sin_2theta_half
    sin_4theta_half = np.sin(4 * theta_half_valid)
    cos_4theta_half = np.cos(4 * theta_half_valid)
    numerator = sin_2theta_half**2
    inner_sqrt = (csc_2theta_half**6) * (sin_4theta_half**2)
    denominator = 4 + np.sqrt(inner_sqrt) - cos_4theta_half * np.sqrt(inner_sqrt)
    result[valid_idx] = 2 * A0 * \
        np.sqrt(np.maximum(numerator / denominator, 0))
    return result


def axis_diff(theta, A0, tol=1e-8):
    """
    True axis diff of the ellipse for a given theta and A0.

    Parameters
    ----------
    theta : array-like
        Differential phase in radians.
    A0 : float
        Amplitude of the underlying signals S+ resp. S-.
    tol : float, optional
        Tolerance for handling singularities at theta = 0, pi. The default is 1e-8.
    
    Returns
    -------
    axis_diff : np.ndarray
        The true axis diff of the ellipse for the given theta and A0.
    """
    theta_half = np.asarray(theta)/2
    theta_half_mod = np.mod(theta_half, np.pi)
    d0 = np.minimum(theta_half_mod, np.pi - theta_half_mod)
    d_pi2 = np.abs(theta_half_mod - np.pi/2)
    mask = (d0 < tol) | (d_pi2 < tol)
    result = np.zeros_like(theta_half)
    result[mask] = np.sqrt(2) * A0
    valid_idx = ~mask
    theta_half_valid = theta_half[valid_idx]
    sin_2theta_half = np.sin(2 * theta_half_valid)
    csc_2theta_half = 1 / sin_2theta_half
    sin_4theta_half = np.sin(4 * theta_half_valid)
    cos_4theta_half = np.cos(4 * theta_half_valid)
    numerator = sin_2theta_half**2
    inner_sqrt = (csc_2theta_half**6) * (sin_4theta_half**2)
    denominator = 4 - np.sqrt(inner_sqrt) + cos_4theta_half * np.sqrt(inner_sqrt)
    result[valid_idx] = 2 * A0 * \
        np.sqrt(np.maximum(numerator / denominator, 0))
    return result


######################################
### conversion between T and theta ###
######################################
def T_calibrated(theta, a, gamma=0.149, tau=100e-6, k=4*np.pi/780.226e-9):
    """
    Calculate the calibrated interferometer time T from the differential phase theta and acceleration a using the formula:
    T = -tau*gamma/2 + sqrt((tau*gamma/2)^2 + theta/(2*a*k)).

    Parameters
    ----------
    theta : array-like
        Differential phase in radians.
    a : array-like
        Measured acceleration in m/s^2.
    gamma : float, optional
        Parameter related to the pulse shape including finite pulse effects.
        The default is 0.149.
    tau : float, optional
        Pulse length in seconds. The default is 100 µs (100e-6 s).
    k : float, optional
        Wavenumber k in 1/m. The default is 4*np.pi/780.226e-9.
    
    Returns
    -------
    T : np.ndarray
        Calibrated interferometer time in seconds corresponding to the given theta and a.
    """
    temp = -tau*gamma/2
    return temp + np.sqrt(temp**2 + theta/(2*a*k))

def parabola_with_linear(t, a, gamma=0.149, tau=100e-6, k=4*np.pi/780.226e-9):
    """
    Theta as a function of time t for an acceleration a, including finite pulse effects, based on the formula:
    theta = 2*k*a*t^2 + 2*k*a*t*tau*gamma.

    Parameters
    ----------
    t : array-like
        Interferometer time in seconds.
    a : array-like
        Measured acceleration in m/s^2.
    gamma : float, optional
        Parameter related to the pulse shape including finite pulse effects.
        The default is 0.149.
    tau : float, optional
        Pulse length in seconds. The default is 100 µs (100e-6 s).
    k : float, optional
        Wavenumber k in 1/m. The default is 4*np.pi/780.226e-9.
    
    Returns
    -------
    theta : np.ndarray
        Differential phase in radians as a function of time t for the given acceleration a, including finite pulse effects.
    """
    return 2*k*a*t**2 + 2*k*a*t*tau*gamma

##################################
### plotting related functions ###
##################################
def get_ax_size_inches(ax, fig):
    """Returns the size of the axis in inches"""
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width_in, height_in = bbox.width, bbox.height
    return width_in, height_in

def plot_line_with_wide_err(ax, x, y, xerr, yerr, color, label=None, elw=1.8, bar_alpha=0.5, marker="+", ls="--"):
    """
    Plot data with custom error bars on a given axis, including options for a connecting line, marker style, and error bar appearance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the data.
    x : array-like
        x-coordinates of the data points.
    y : array-like
        y-coordinates of the data points.
    xerr : array-like
        Errors in x-coordinates (can be symmetric or asymmetric).
    yerr : array-like
        Errors in y-coordinates (can be symmetric or asymmetric).
    color : str or tuple
        Colour for the markers and error bars.
    label : str, optional
        Label for the data series (used in legend). The default is None (no label).
    elw : float, optional
        Line width for the error bars. The default is 1.8.
    bar_alpha : float, optional
        Transparency for the error bars (between 0 and 1). The default is 0.5.
    marker : str, optional
        Marker style for the data points (e.g. '+', 'o', 's'). The default is '+'.
    ls : str, optional
        Line style for the connecting line (e.g. '--', '-', ':'). The default is '--'.
    """
    # connecting line
    ax.plot(x, y, color=color, linewidth=0.3, alpha=0.5, linestyle=ls)

    # markers + error bars (no caps, thicker bars)
    ln, caplines, barcols = ax.errorbar(
        x, y,
        xerr=xerr,
        yerr=yerr,
        fmt=marker,            # marker at data points
        linestyle='',          # avoid a second connecting line from errorbar
        color=color,           # marker color
        ecolor=color,          # error bar color
        elinewidth=elw,        # thickness of the vertical bars
        capsize=0,             # no caps
        label=label
    )
    # set transparency only on the error bar LineCollection(s)
    # barcols is a tuple of LineCollections (horizontal, vertical); set both
    if isinstance(barcols, (list, tuple)):
        for lc in barcols:
            lc.set_alpha(bar_alpha)
    else:
        barcols.set_alpha(bar_alpha)
