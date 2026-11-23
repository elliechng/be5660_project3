"""Paper comparisons from in-memory scores and covariates."""
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from regression import (ENDPOINTS, STAGES, design, disease_extra, stage_design,
                        outcome_extra, fit_hc3, group_counts, holm_pair)


def fit_endpoint(frame, endpoint, extra=None, stage='M1'):
    return fit_hc3(frame[endpoint], design(frame, extra, stage), term='patient')[0]


def primary_comparisons(frame):
    """Primary tests, separate-center estimates, staged and reduced adjustment."""
    primary, sites, stages = [], [], []
    for endpoint in ENDPOINTS:
        row = fit_endpoint(frame, endpoint, disease_extra(frame, endpoint))
        primary.append(dict(endpoint=endpoint, **row, **group_counts(frame)))
        for site, sub in frame.groupby('SITE', sort=True):
            row = fit_endpoint(sub, endpoint, disease_extra(sub, endpoint))
            sites.append(dict(endpoint=endpoint, SITE=site, **row, **group_counts(sub)))
    primary = pd.DataFrame(primary)
    primary['holm_p'] = holm_pair(primary.hc3_p)
    for stage in STAGES:
        row, _ = fit_hc3(frame.resilience_gap, stage_design(frame, stage), term='patient')
        stages.append(dict(stage=stage, **row, **group_counts(frame)))
    reduced = {
        'unadjusted_hub': fit_endpoint(frame, ENDPOINTS[0], stage='M0'),
        'hub_without_overall_FA_adjustment': fit_endpoint(
            frame, ENDPOINTS[0], {'FA_coverage': frame.FA_coverage}),
        'mean_absolute_FA_by_group': frame.groupby('isControl').mean_absolute_fa_z.mean(),
        'raw_covariates_by_group': frame.groupby('isControl')[['raw_log_strength', 'raw_density']].mean(),
    }
    return primary, pd.DataFrame(sites), pd.DataFrame(stages), reduced


def outcome_comparisons(scores):
    """Two score associations per cohort, with separate Holm families."""
    rows = []
    for subset in ['expanded_reviewed', 'original_one_year']:
        sub = scores.copy()
        if subset == 'original_one_year':
            sub = sub.loc[sub.outcome_source.eq('audited_one_year')].copy()
        # Reuse the group model with Engel I as the reference.
        sub['isControl'] = sub.outcome.eq(0)
        for endpoint in ENDPOINTS:
            row = fit_endpoint(sub, endpoint, outcome_extra(sub))
            rows.append(dict(subset=subset, endpoint=endpoint,
                             Engel_I=int(sub.isControl.sum()),
                             Engel_II_IV=int((~sub.isControl).sum()), **row))
    result = pd.DataFrame(rows)
    for subset in result.subset.unique():
        keep = result.subset.eq(subset)
        result.loc[keep, 'holm_p'] = holm_pair(result.loc[keep, 'hc3_p'])
    return result


def resilience_sensitivities(frame, density_frames):
    """Overlap restriction, leave-one-center-out and alternative densities."""
    keep = pd.Series(True, index=frame.index)
    # Define overlap within each center before pooling the retained rows.
    for _, sub in frame.groupby('SITE'):
        for name in ['age', 'raw_density', 'raw_log_strength', 'baseline_efficiency']:
            groups = [sub.loc[sub.isControl.eq(g), name] for g in [False, True]]
            low = max(values.quantile(.05) for values in groups)
            high = min(values.quantile(.95) for values in groups)
            keep.loc[sub.index] &= sub[name].between(low, high)
    subsets = [('Central overlap', frame.loc[keep])]
    subsets += [('Without ' + site, frame.loc[frame.SITE.ne(site)])
                for site in sorted(frame.SITE.unique())]
    subsets += [(f'{density:.0%} density', sub) for density, sub in density_frames.items()]
    rows = []
    for label, sub in subsets:
        row, _ = fit_hc3(sub.resilience_gap, stage_design(sub, 'M3'), term='patient')
        rows.append(dict(check=label, **row, **group_counts(sub)))
    return pd.DataFrame(rows)


def attack_curve_differences(curves, controls, budgets, n_nodes=82):
    """Unadjusted patient-minus-control HC3 estimates at each removal level."""
    curves = np.asarray(curves, float)
    controls = np.asarray(controls, bool)
    x = np.asarray(budgets, float) / n_nodes * 100
    patient = (~controls).astype(float)
    design_matrix = sm.add_constant(patient)
    rows = []
    for label, values in [('Targeted', curves[:, 0]),
                          ('Random', curves[:, 1:].mean(axis=1))]:
        for k in range(len(x)):
            # At zero removal, every normalized curve starts at one.
            if k == 0:
                beta, se = 0., 0.
            else:
                fit = sm.OLS(values[:, k], design_matrix).fit(cov_type='HC3', use_t=False)
                beta, se = fit.params[1], fit.bse[1]
            low, high = beta + norm.ppf([.025, .975]) * se
            rows.append(dict(attack=label, percent_isolated=x[k], beta=beta, hc3_se=se,
                             hc3_low=low, hc3_high=high, patients=int(patient.sum()),
                             controls=int((1-patient).sum())))
    return pd.DataFrame(rows)
