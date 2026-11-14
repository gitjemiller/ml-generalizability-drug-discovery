# ML Generalizability Across Chemical Space

Independent research project on how molecular property prediction models degrade on scaffolds they have not seen, and whether a standard applicability domain (AD) score can predict where they fail.

## Research question

How much does a random-split evaluation overstate model performance relative to a scaffold split, and does k-NN Tanimoto similarity to the training set predict prediction error once scaffolds are held out?

## Scope

This is a single-endpoint study on ESOL (aqueous solubility, log S). The original plan included BBBP, CYP3A4, a GCN baseline, and a Butina cluster split. Those were not run and are listed under Future work.

## Method

- **Data:** ESOL, loaded from CSV. Acyclic compounds were dropped because they have no Bemis-Murcko scaffold (28% of the dataset), leaving n = 811 compounds and 268 unique scaffolds.
- **Features:** Morgan fingerprints (ECFP4, radius 2, 2048 bits) via RDKit.
- **Models:** Random Forest and XGBoost, trained on identical split indices.
- **Splits:** random 80/20 and Bemis-Murcko scaffold 80/20. The scaffold split uses a size-aware greedy fill: scaffold groups are shuffled, then assigned to test until the test fraction is reached. This keeps test size stable across seeds with zero scaffold overlap.
- **Seeds:** five per split.
- **AD score:** mean Tanimoto similarity to the k = 5 nearest training compounds.

## Results

Random Forest test RMSE (log S), mean ± SD over five seeds:

| Split | Test RMSE |
| --- | --- |
| Random | 1.191 ± 0.092 |
| Scaffold | 2.040 ± 0.069 |

RMSE by AD quartile (Q1 = least similar to training set, Q4 = most similar):

| Split | Q1 | Q2 | Q3 | Q4 |
| --- | --- | --- | --- | --- |
| Random | 1.38 | 1.46 | 1.15 | 0.97 |
| Scaffold | 1.53 | 2.16 | 2.41 | 2.23 |

Full per-seed numbers are in `results/`. Figures are in `figures/`.

## Finding

Under random split, error falls as similarity to the training set rises, as expected. Under scaffold split, the AD score appears confounded with target value and k-NN Tanimoto similarity fails in error prediction, even inverting in all five seeds. In this dataset, highly familiar scaffolds happened to be the poorly soluble ones. It should be noted though, structural novelty is still a factor.

Molecular weight was checked as a confound and ruled out (`check_confound.py`). The mean target value runs from -3.08 to -5.17 log S across AD quartiles, which identifies the target distribution as the mechanism.

## Limitations

- One dataset and one endpoint. The inversion may be specific to ESOL.
- Fingerprint-based models only. No graph neural network baseline.
- One AD method.
- An earlier Week 1 baseline reported test RMSE 0.799. That run included acyclic compounds, used a different data path, and recorded no seed. The numbers above supersede it.

## Future work

Additional endpoints (BBBP, CYP3A4), a GCN baseline, Butina cluster splits, and AD methods that account for the target distribution.

## Repository structure

```
├── src/                  # data loading, features, splits, AD methods
├── run_experiment.py     # random vs. scaffold split, RF and XGBoost, five seeds
├── run_ad_analysis.py    # AD score and RMSE by quartile
├── check_split.py        # split sizes and scaffold overlap
├── check_scaffolds.py    # scaffold counts
├── check_confound.py     # molecular weight vs. target value confound check
├── results/              # per-seed outputs
├── figures/
├── notebooks/            # Week 1 baseline notebook
├── data/
└── research_log.md
```

## Reproducing

```bash
conda env create -f environment.yml
conda activate drugdisc
python run_experiment.py
python run_ad_analysis.py
```

## Status

Complete as a single-endpoint study (August 2026).
