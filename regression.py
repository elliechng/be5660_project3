"""OLS group differences with HC3 intervals and separate two-test Holm families."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests


def fit_hc3(y, x, term='group'):
    y = np.asarray(y, float)
    design_values = x.to_numpy(float)
    assert np.isfinite(y).all() and np.isfinite(design_values).all()
    assert np.linalg.matrix_rank(design_values) == design_values.shape[1], 'Rank-deficient model'
    # HC3 needs residual information beyond a single high-leverage row.
    leverage = (design_values * np.linalg.pinv(design_values).T).sum(axis=1)
    assert leverage.max() < 1 - 1e-8, 'Unit leverage prevents HC3 inference'
    fit = sm.OLS(y, x).fit(cov_type='HC3', use_t=False)
    low, high = fit.conf_int(alpha=.05).loc[term]
    beta, se = float(fit.params[term]), float(fit.bse[term])

    result = dict(n=len(y), beta=beta, hc3_se=se, hc3_low=float(low),
                  hc3_high=float(high), hc3_p=float(fit.pvalues[term]))
    coefficients = pd.DataFrame({'term': x.columns, 'coefficient': fit.params.to_numpy(),
                                 'HC3_SE': fit.bse.to_numpy(), 'HC3_p': fit.pvalues.to_numpy()})
    return result, coefficients


def holm_pair(p_values):
    """Correct the two tests within one prespecified comparison family."""
    values = np.asarray(p_values, float)
    if values.shape != (2,) or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Expected two finite p-values between zero and one')
    return multipletests(values, method='holm')[1]


ENDPOINTS = ('excess_hub_concentration', 'resilience_gap')
STAGES = ('M0', 'M1', 'M2', 'M3')


def design(frame, extra=None, stage='M1'):
    """Build the group, age, sex and site columns used in the paper."""
    x = pd.DataFrame({'patient': (~frame.isControl).astype(float),
                      'intercept': 1.}, index=frame.index)
    if stage != 'M0':
        x['age_decade'] = frame.age.astype(float) / 10
        x['sex'] = frame.sexisMale.astype(float)
        for site, values in pd.get_dummies(frame.SITE, drop_first=True, dtype=float).items():
            x['SITE_' + site] = values
    if extra:
        for name, values in extra.items():
            values = np.asarray(values, float)
            if np.ptp(values) > 1e-12:
                x[name] = values
    return x.astype(float)


def disease_extra(frame, endpoint):
    names = (['FA_coverage', 'mean_absolute_fa_z'] if endpoint == ENDPOINTS[0]
             else ['raw_density', 'raw_log_strength', 'baseline_efficiency'])
    return {name: frame[name] for name in names}


def stage_design(frame, stage):
    extra = {}
    if stage in ('M2', 'M3'):
        extra.update({name: frame[name] for name in ['raw_density', 'raw_log_strength']})
    if stage == 'M3':
        extra['baseline_efficiency'] = frame.baseline_efficiency
    return design(frame, extra, stage)


def outcome_extra(frame):
    extra = dict(resection=frame.procedure.eq('resection').astype(float),
                 measured_edge_coverage=frame.measured_edge_coverage,
                 original_extra_mesial_FA=frame.extra_mesial_mean_abs_fa_z,
                 EpilepsyDuration=frame.EpilepsyDuration.fillna(frame.EpilepsyDuration.median()),
                 EpilepsyDuration_missing=frame.EpilepsyDuration.isna().astype(float))
    # Keep unknown MTS separate from documented absence.
    for name in ['present', 'unknown']:
        extra['MTS_' + name] = frame.MTS.eq(name).astype(float)
    return extra


def group_counts(frame):
    return dict(patients=int((~frame.isControl).sum()), controls=int(frame.isControl.sum()))
