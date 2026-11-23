"""Control-reference hubs, spatial concentration and structural network attacks."""
import numpy as np
from scipy.sparse.csgraph import floyd_warshall


def control_hubness(networks, metadata, training=None):
    """Equal-site mean of relative node strengths; patients never enter."""
    controls = metadata.isControl.to_numpy(bool).copy()
    if training is not None:
        controls &= np.asarray(training, bool)
    strength = np.asarray(networks).sum(axis=2)
    relative = strength / strength.sum(axis=1, keepdims=True)
    site_profiles = []
    for site in sorted(metadata.SITE.unique()):
        selected = controls & metadata.SITE.eq(site).to_numpy()
        if not selected.any():
            raise ValueError('Missing training controls at a site')
        site_profiles.append(relative[selected].mean(axis=0))
    # Give each center equal weight, regardless of its number of controls.
    profile = np.mean(site_profiles, axis=0)
    if profile.std() <= 1e-12:
        raise ValueError('Control hubness is constant')
    return (profile - profile.mean()) / profile.std()


def geometry_strata(distance, hemisphere, reference, bins=10):
    i, j = np.triu_indices(len(hemisphere), 1)
    lengths = np.asarray(distance)[i, j]
    cuts = np.unique(np.quantile(lengths[np.asarray(reference, bool)], np.linspace(0, 1, bins + 1)))
    distance_bin = np.searchsorted(cuts[1:-1], lengths, side='right')
    hemisphere = np.asarray(hemisphere)
    pair = np.where(hemisphere[i] != hemisphere[j], 1, np.where(hemisphere[i] == 'L', 0, 2))
    return distance_bin * 3 + pair, cuts


def hub_concentration(absolute_z, edge_hubness, strata):
    """Observed and spatially matched hub concentration for one participant."""
    a, h, strata = map(np.asarray, (absolute_z, edge_hubness, strata))
    if not (a.shape == h.shape == strata.shape):
        raise ValueError('Edge vectors disagree')
    good = np.isfinite(a) & np.isfinite(h)
    a, h, strata = a[good].astype(float), h[good].astype(float), strata[good]
    if (a < 0).any():
        raise ValueError('Absolute deviations cannot be negative')
    total = a.sum()
    if not len(a) or total <= 1e-12:
        return dict(eligible=False, status='zero_absolute_deviation_sum')
    expected, variance = 0., 0.
    for label in np.unique(strata):
        aa, hh = a[strata == label], h[strata == label]
        expected += aa.mean() * hh.sum()
        if len(aa) > 1:
            component = ((aa - aa.mean()) ** 2).sum() * ((hh - hh.mean()) ** 2).sum() / (len(aa) - 1)
            variance += component
    observed = float(a @ h)
    excess = (observed - expected) / total
    return dict(eligible=bool(variance > 1e-12), status='eligible' if variance > 1e-12 else 'zero_null_variance',
        excess_hub_concentration=float(excess), observed_concentration=observed / total,
        expected_concentration=float(expected / total), measured_edges=len(a))


def efficiency_from_lengths(lengths, remaining=None):
    """Removed/unreachable pairs contribute zero; denominator stays at original N."""
    lengths = np.asarray(lengths, float)
    # Keep this denominator after nodes are isolated.
    original_n = len(lengths)
    if remaining is not None:
        lengths = lengths[np.ix_(remaining, remaining)]
    if len(lengths) < 2:
        return 0.
    distances = floyd_warshall(lengths, directed=False)
    inverse = np.divide(1., distances, out=np.zeros_like(distances),
                        where=np.isfinite(distances) & (distances > 0))
    return float(inverse.sum() / (original_n * (original_n - 1)))


def attack_summary(weights, targeted_order, random_orders, budgets):
    weights = np.asarray(weights, float)
    n = len(weights)
    targeted_order, random_orders, budgets = map(np.asarray, (targeted_order, random_orders, budgets))
    if (weights.shape != (n, n) or (weights < 0).any() or not np.isfinite(weights).all()
        or not np.allclose(weights, weights.T)):
        raise ValueError('Invalid attack network')
    if not np.array_equal(np.sort(targeted_order), np.arange(n)):
        raise ValueError('Target order must include each node once')
    if random_orders.ndim != 2 or random_orders.shape[1] != n or not np.all(np.sort(random_orders, axis=1) == np.arange(n)):
        raise ValueError('Random orders must be full node permutations')
    if len(random_orders) < 2 or budgets[0] != 0 or (np.diff(budgets) <= 0).any() or budgets[-1] >= n:
        raise ValueError('Invalid attack budgets or replicate count')
    lengths = np.divide(1., weights, out=np.full_like(weights, np.inf), where=weights > 0)
    np.fill_diagonal(lengths, 0)
    baseline = efficiency_from_lengths(lengths)
    if baseline <= 0:
        raise ValueError('Unperturbed graph has zero efficiency')
    orders = np.vstack([targeted_order, random_orders])
    curves = np.ones((len(orders), len(budgets)))
    for row, order in enumerate(orders):
        for column, budget in enumerate(budgets[1:], 1):
            curves[row, column] = efficiency_from_lengths(lengths, order[budget:]) / baseline
    fractions = budgets / n
    differences = np.trapezoid(curves[1:] - curves[0], fractions, axis=1) / fractions[-1]
    return dict(resilience_gap=float(differences.mean()), baseline_efficiency=baseline), curves
