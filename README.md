# Hub concentration and structural network resilience

Analysis and figure code for *Hub involvement and structural network resilience in temporal lobe epilepsy*.

Repository: https://github.com/elliechng/be5660_project3

## Files

- `fa_reference.py`: measured-FA masks, site-specific control support and control-only Huber reference fits. A training mask excludes held-out controls.
- `hub_resilience.py`: control-reference hubness, distance/hemisphere strata, spatially matched FA concentration, global efficiency and targeted-versus-random isolation.
- `regression.py`: the paper's design matrices, clinical covariate coding, OLS/HC3 inference and two-test Holm correction.
- `analysis.py`: primary comparisons, reduced and staged adjustment, separate-center estimates, outcome associations, the one-year subset, appendix sensitivities and attack-curve confidence bands.
- `figures.py`: the schematic and all four results figures, including the cortical surface rendering.
- `requirements.txt`: Python dependencies.

## Data availability

No participant data, notes, identifiers, fitted reference objects, coefficients, score tables, images or results are distributed here. The datasets generated and/or analyzed during this study are not publicly available due to patient privacy regulations and institutional review board restrictions, as they contain protected health information derived from institutional electronic health records.

Data access, clinical linkage, older project utilities and analyses outside the paper are excluded. This code release requires authorized inputs and does not include the complete private data-preparation workflow.
