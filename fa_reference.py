"""Control-only FA deviations on measured, reference-evaluable connections."""
import numpy as np
import pandas as pd
import statsmodels.api as sm


def measured_fa(count, fa):
    """A missing tract is not an observed zero FA; zero is valid if measured."""
    count, fa = np.asarray(count), np.asarray(fa)
    if count.shape != fa.shape:
        raise ValueError('FA/count shapes disagree')
    return np.isfinite(count) & (count > 0) & np.isfinite(fa) & (fa >= 0) & (fa <= 1)


def control_support(measured, metadata, training=None, minimum=.5):
    """Intersect site-specific support masks using training controls only."""
    controls = metadata.isControl.astype(bool).to_numpy()
    if training is not None:
        controls = controls & np.asarray(training, bool)
    support = np.ones(measured.shape[1], bool)
    for site in sorted(metadata.SITE.unique()):
        keep = controls & (metadata.SITE.to_numpy() == site)
        if not keep.any():
            raise ValueError(f'No training controls at {site}')
        support &= measured[keep].mean(axis=0) >= minimum
    return support


def observed_fa_reference(count, fa, metadata, training=None, threshold=3.5):
    """Fit edgewise control references; inputs are participant-by-edge arrays."""
    measured = measured_fa(count, fa)
    controls = metadata.isControl.astype(bool).to_numpy()
    if training is not None:
        controls = controls & np.asarray(training, bool)
    # Rebuild support as well as the fits when a control fold is held out.
    support = control_support(measured, metadata, training)
    sites = pd.Categorical(metadata.SITE, categories=sorted(metadata.SITE.unique()))
    site_indicators = pd.get_dummies(sites, drop_first=True, dtype=float).to_numpy()
    design_matrix = np.column_stack([np.ones(len(metadata)), metadata[['age', 'sexisMale']].to_numpy(float), site_indicators])
    if not np.isfinite(design_matrix).all():
        raise ValueError('Missing normative covariates')
    z = np.full(fa.shape, np.nan)
    coefficients = np.full((fa.shape[1], design_matrix.shape[1]), np.nan)
    records = []
    for edge in range(fa.shape[1]):
        keep = controls & measured[:, edge]
        status, mu, sigma, iterations, capped_stable = 'kept', np.nan, np.nan, 0, False
        n = int(keep.sum())
        if not support[edge]:
            status = 'insufficient_site_support'
        elif np.linalg.matrix_rank(design_matrix[keep]) < design_matrix.shape[1]:
            status = 'rank_deficient_reference'
        elif np.std(fa[keep, edge]) <= 1e-12:
            status = 'constant_reference'
        else:
            def fit_score(maxiter):
                fit = sm.RLM(fa[keep, edge], design_matrix[keep], M=sm.robust.norms.HuberT()).fit(maxiter=maxiter)
                residual = fa[:, edge] - design_matrix @ fit.params
                control_residuals = residual[keep]
                rough_sd = control_residuals.std()
                clean = control_residuals[np.abs((control_residuals - control_residuals.mean()) / rough_sd) < 3] if rough_sd > 1e-12 else np.array([])
                if len(clean) < 2 or not np.isfinite(clean.std()) or clean.std() <= 1e-12:
                    return fit, None, np.nan, np.nan
                return fit, (residual - clean.mean()) / clean.std(), float(clean.mean()), float(clean.std())
            fit, score, mu, sigma = fit_score(500)
            iterations = int(fit.fit_history['iteration'])
            if score is None:
                status = 'zero_residual_variance'
            else:
                # Check score stability if the initial fit reaches its limit.
                if iterations >= 500:
                    _, later, _, _ = fit_score(1000)

                    valid = measured[:, edge] & np.isfinite(score)
                    if later is None or not np.array_equal(np.abs(score[valid]) >= threshold,
                                                           np.abs(later[valid]) >= threshold):
                        raise ValueError(f'Unstable normative classification at edge {edge}')
                    if np.max(np.abs(score[valid] - later[valid])) > .01:
                        raise ValueError(f'Unstable continuous normative score at edge {edge}')
                    capped_stable = True
                z[measured[:, edge], edge] = score[measured[:, edge]]
                coefficients[edge] = fit.params
        records.append((edge, bool(support[edge]), status, n, mu, sigma, iterations, capped_stable))
    diagnostic = pd.DataFrame(records, columns=['edge', 'candidate_support', 'status',
        'measured_training_controls', 'reference_mean', 'reference_sd', 'iterations', 'capped_stable'])
    diagnostic['evaluated'] = diagnostic.status == 'kept'
    return z, diagnostic, coefficients
