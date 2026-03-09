# Data Directory

## Policy
Raw data files are NOT committed to this repository. 
All datasets are loaded programmatically at runtime via DeepChem MoleculeNet.

## Datasets Used

| Dataset | Source | DeepChem Loader |
|---------|--------|-----------------|
| ESOL | Delaney (2004), MoleculeNet | dc.molnet.load_delaney() |
| BBBP | Martins et al. (2012), MoleculeNet | dc.molnet.load_bbbp() |
| CYP3A4 | Veith et al. (2009), MoleculeNet | dc.molnet.load_cyp3a4_substrate_labeling() |

## Notes
- All datasets accessed via MoleculeNet benchmark suite
- No manual download required
- DeepChem caches downloads locally on first run