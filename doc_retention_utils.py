import numpy as np
import lmfit
from lmfit import Model
from scipy.stats import chi2
from sklearn.metrics import r2_score, mean_absolute_error


def transmission_k_constant(tau, k=0.5):
    """Model assuming exponential decay with constant k"""
    return np.exp(-k * tau)


def transmission_k_as_func_of_tau(tau, a=0.5, b=-0.5):
    """Model assuming exponential decay with k=f(t)"""
    return np.exp(-a * (tau ** (b + 1)))


def transmission_sigma_constant(tau, sigma=0.5):
    """Vollenweider model, constant sigma"""
    return 1 / (1 + (sigma * tau))


def transmission_sigma_as_func_of_tau(tau, a=0.5, b=-0.5):
    """Vollenweider model, sigma=f(tau)"""
    return 1 / (1 + (a * (tau ** (1 + b))))


def fit_model(func, df, tau_col, trans_col):
    """Convenience function for fitting models and priting summary statistics.

    Args
        func:      Function to be fit
        df:        Dataframe. Data to fit
        tau_col:   Str. Name of residence time column in 'df'
        trans_col: Str. Name of transmission column in 'df'

    Returns
        Model result object from lmfit. Summary stats. are printed.
    """
    model = Model(func, independent_vars=["tau"])
    fit = model.fit(df[trans_col], tau=df[tau_col])
    print(fit.fit_report())
    r2 = 1 - fit.residual.var() / np.var(df[trans_col])
    print(f"R2: {r2:.2f}")

    return fit


def likelihood_ratio_test(model1, model2):
    """
    Perform a likelihood ratio test between two models.

    This function calculates the likelihood ratio test statistic and p-value to compare
    the goodness of fit between two nested models. It prints the likelihood ratio, degrees
    of freedom, and p-value.

    Parameters:
    model1 (lmfit.model.ModelResult): The first (simpler) model.
    model2 (lmfit.model.ModelResult): The second (more complex) model.

    Returns:
    None
    """
    chi_square1 = model1.result.chisqr
    chi_square2 = model2.result.chisqr

    n = len(model1.data)

    # Calculate the log-likelihood
    log_likelihood1 = -0.5 * chi_square1
    log_likelihood2 = -0.5 * chi_square2

    # Calculate the test statistic
    LR = -2 * (log_likelihood1 - log_likelihood2)

    # Degrees of freedom
    deg_freedom = model2.nvarys - model1.nvarys

    # Calculate the p-value
    p_value = chi2.sf(LR, deg_freedom)

    print(f"Likelihood Ratio: {LR}")
    print(f"Degrees of Freedom: {deg_freedom}")
    print(f"P-value: {round(p_value, 3)}")


def calculate_mae_and_mbd(model, data, x_col, y_col):
    # Extract the observed and predicted values
    observed = data[y_col].values
    predicted = model.eval(params=model.params, **{x_col: data[x_col].values})

    # Ensure the lengths of observed and predicted are the same
    if len(observed) != len(predicted):
        raise ValueError(f"Length mismatch: observed has {len(observed)} elements, predicted has {len(predicted)} elements")

    # Calculate the errors
    errors = observed - predicted

    # Calculate the mean absolute error (MAE)
    mae = np.mean(np.abs(errors))

    # Calculate the mean bias deviation (MBD)
    mbd = np.mean(errors)

    return mae, mbd


def rs_mae(s1_obs, s2_pred, model_name):
    r2 = r2_score(s1_obs, s2_pred)
    mae = mean_absolute_error(s1_obs, s2_pred)
    print(f"{model_name}:")
    print("R2:", np.round(r2, 2), "MAE:", np.round(mae, 2))


def calculate_bic(obs_s, pred_s, num_params):
    """
    Calculate Bayesian Information Criterion (BIC).

    Parameters:
    df (pd.DataFrame): DataFrame containing the data.
    obs_col (str): Column name for observations.
    pred_col (str): Column name for predictions.
    num_params (int): Number of parameters in the model.

    Returns:
    float: BIC value.
    """
    n = len(obs_s)
    residual_sum_of_squares = np.sum((obs_s - pred_s) ** 2)
    bic = n * np.log(residual_sum_of_squares / n) + num_params * np.log(n)
    print(bic)


def st_dev_residuals(model_obj):
    # Calculate residual standard deviation
    residuals = model_obj.residual
    residual_std = np.std(residuals)
    print("Standard deviation of residuals:", np.round(residual_std, 3))
    return residual_std


def generate_prediction_intervals(model_obj, x, n_samples=1000, add_residual_error=True):
    """
    To do: generalise to remove hard-coding of param names
    """
    params = model_obj.params

    if len(params) < 2:
        # Handle single parameter case
        y_pred = transmission_sigma_constant(x, **params.valuesdict())

        k = params['sigma'].value
        k_stderr = params['sigma'].stderr
        if k_stderr is None:
            raise ValueError("Standard error for parameter 'k' is not available.")

        # Create confidence intervals manually
        # Assuming a normal distribution (mean=0, s.d.=1), 95% of data are between +/-1.96
        ci_k_l95 = k - 1.96 * k_stderr
        ci_k_u95 = k + 1.96 * k_stderr

        y_samples = np.zeros((n_samples, len(x)))
        for i in range(n_samples):
            k_sample = np.random.uniform(ci_k_l95, ci_k_u95)
            y_samples[i, :] = transmission_sigma_constant(x, sigma=k_sample)

        y_upper = np.max(y_samples, axis=0)
        y_lower = np.min(y_samples, axis=0)
    else:
        # Handle multiple parameter case
        y_pred = transmission_sigma_as_func_of_tau(x, **params.valuesdict())
        ci = lmfit.conf_interval(model_obj, result=model_obj)
        ci_k_l95 = ci['k'][1][1]
        ci_k_u95 = ci['k'][-2][1]
        ci_m_l95 = ci['m'][1][1]
        ci_m_u95 = ci['m'][-2][1]

        y_samples = np.zeros((n_samples, len(x)))
        for i in range(n_samples):
            k_sample = np.random.uniform(ci_k_l95, ci_k_u95)
            m_sample = np.random.uniform(ci_m_l95, ci_m_u95)
            y_samples[i, :] = transmission_sigma_as_func_of_tau(x, k=k_sample, m=m_sample)

        y_upper = np.max(y_samples, axis=0)
        y_lower = np.min(y_samples, axis=0)

    if add_residual_error:
        # Calculate residual standard deviation
        residual_std = st_dev_residuals(model_obj)

        # Adjust intervals to include 95% CIs on residual error
        y_upper += 1.96 * residual_std
        y_lower -= 1.96 * residual_std

    return y_pred, y_upper, y_lower