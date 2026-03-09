# Research Log — ML Generalizability Drug Discovery

## Purpose
This log tracks weekly progress, decisions made, problems encountered, 
and open questions. Written for my future self and potential collaborators.

---

## Week 1 — Environment & First Notebook

**Date:** 03/08/2026

**What I did:**
- Set up conda environment (drugdisc, Python 3.10)
- Installed DeepChem, RDKit, scikit-learn, XGBoost, UMAP, JupyterLab
- Initialized GitHub repo with project brief folder structure
- Loaded ESOL dataset via DeepChem
- Computed Morgan fingerprints (ECFP4, radius=2, 2048 bits)
- Trained RandomForestRegressor, reported RMSE

**Results:**
- Training RMSE:  0.226 log S
- Test RMSE: 0.799 log S

**Generalization gap:** 0.573 log S (test - train)
This is expected under random split, the test set contains molecules 
structurally similar to training data. This is our performance ceiling. 
Scaffold and cluster splits will stress-test whether this holds.

**What confused me:**
Honestly, jumping back into the computational chem world and taking it up a notch is a bit jarring. Some things are like riding a bike, others I don't remember or are completely new. Morgan Fingerprints make sense especially given my background with Dr. DeYonker and SMILES I remember from undergrad. Adding the ML component has brought an interesting touch. RMSE is how much your model is off normally and is different from R^2.

**Open questions:**
What's the difference between Test and Training RMSE?
What is log S?
I'd love to learn more about the RandomForestRegressor
How would the 80/20 split we talked about affect the RMSE and training accuracy?

---