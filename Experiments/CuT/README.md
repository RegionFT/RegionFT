# Classifiers under Test

This directory contains the 16 classifiers used in the ICSE 2027 experiments:
four model families trained on each of the four datasets.

- Models: `GBDT`, `MLP`, `LogReg`, and `DecTree`
- Datasets: `Adult`, `Credit`, `Bank`, and `Lsac`
- Naming convention: `<Model><Dataset>.joblib`

[`TrainClassifierUnderTest.py`](TrainClassifierUnderTest.py) reproduces the
training process with a fixed random seed. [`test_acc_log.json`](test_acc_log.json)
records held-out test metrics for the saved models.
