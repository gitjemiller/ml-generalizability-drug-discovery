# ML Generalizability Across Chemical Space

**Independent research project evaluating how ML model performance degrades
across increasingly dissimilar chemical space.**

## Research Question
How does ML model performance degrade when evaluated under increasingly
stringent splitting regimes, and what molecular features predict when
applicability domain failures are recoverable?

## Datasets
| Dataset | Endpoint | Task | Size |
|---------|----------|------|------|
| ESOL | Aqueous solubility (log S) | Regression | ~1,100 |
| BBBP | Blood-brain barrier penetration | Classification | ~2,000 |
| CYP3A4 | CYP3A4 inhibition | Classification | ~12,000 |

## Methods
- **Features:** Morgan fingerprints (ECFP4, radius=2, 2048 bits) via RDKit
- **Models:** Random Forest, XGBoost, GCN (DeepChem)
- **Splits:** Random (80/20), Scaffold (Bemis-Murcko), Butina cluster
- **AD Method:** k-NN Tanimoto distance (k=5)

## Repository Structure
```
ml-generalizability-drug-discovery/
├── notebooks/
│   ├── 00_overview.ipynb          # Visual abstract — start here
│   ├── 01_baseline_models.ipynb
│   ├── 02_scaffold_analysis.ipynb
│   └── 03_applicability_domain.ipynb
├── src/
│   ├── splits.py
│   ├── features.py
│   └── ad_methods.py
├── figures/
├── data/README.md
└── research_log.md
```

## Reproducing This Project
```bash
conda env create -f environment.yml
conda activate drugdisc
jupyter lab
```

## Status
🔬 Complete — Week 1 (March 2025)
