import numpy as np
from lmfit import Model
from scipy.stats import chi2
from lmfit import Model
from sklearn.metrics import r2_score, mean_absolute_error


def transmission_k_constant(tau, k=0.5):
    """Model assuming exponential decay with constant k"""
    return np.exp(-k * tau)


def transmission_k_as_func_of_tau(tau, p=0.5, m=-0.5):
    """Model assuming exponential decay with k=f(t)"""
    return np.exp(-p * (tau ** (m + 1)))


def transmission_sigma_constant(tau, sigma=0.5):
    """Vollenweider model, constant sigma"""
    return 1 / (1 + (sigma * tau))


def transmission_sigma_as_func_of_tau(tau, k=0.5, m=-0.5):
    """Vollenweider model, sigma=f(tau)"""
    return 1 / (1 + (k * (tau ** (1 + m))))


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