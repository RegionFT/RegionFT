# Datasets

This directory contains the four preprocessed tabular datasets used in the
evaluation.

| Experiment name | File | Features | Protected attributes | Source/reference |
|---|---|---:|---|---|
| Adult | [`Adult.csv`](Adult.csv) | 13 | sex, race, age | [UCI Adult](https://doi.org/10.24432/C5XW20) |
| Credit | [`GermanCredit.csv`](GermanCredit.csv) | 20 | sex, age | [UCI German Credit](https://doi.org/10.24432/C5NC77) |
| Bank | [`Bank.csv`](Bank.csv) | 16 | age | [UCI Bank Marketing](https://doi.org/10.24432/C5K306) |
| Lsac | [`Lsac.csv`](Lsac.csv) | 11 | sex, race | [LSAC National Longitudinal Bar Passage Study](https://eric.ed.gov/?id=ED469370) |

Following common practice in fairness-testing research on tabular data, the
datasets are preprocessed into integer-valued CSV files. The experiment and
classifier-training code use these files directly. The final column is the
classification label; rows with missing values were removed from Adult during
preprocessing.
