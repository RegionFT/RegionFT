# Baseline Implementations

### Evaluated in RQ1

RQ1 compares RegionFT with five representative individual fairness testing
methods.

| Method | Paper | Implementation |
| --- | --- | --- |
| AFT | <https://doi.org/10.1145/3691620.3695481> | <https://github.com/toda-lab/AFT> |
| ExpGA | <https://doi.org/10.1145/3510003.3510137> | <https://github.com/waving7799/ExpGA> |
| GRFT | <https://doi.org/10.1109/ICSE55347.2025.00235> | <https://anonymous.4open.science/r/GRFT-8C8A> |
| LIMI | <https://doi.org/10.1145/3597926.3598099> | <https://github.com/xiaoyisong/Latent_Imitator> |
| SG | <https://doi.org/10.1145/3338906.3338937> | <https://github.com/pxzhang94/ADF> |

### Also supported by `runner.py`

`runner.py` also supports three methods that are not part of the RQ1 comparison.

| Method | Paper | Implementation |
| --- | --- | --- |
| Themis | <https://doi.org/10.1145/3106237.3106277> | Implemented in the AFT study: <https://github.com/toda-lab/AFT> |
| VBT | <https://doi.org/10.1007/978-3-030-64881-7_16> | Implemented in the AFT study: <https://github.com/toda-lab/AFT> |
| VBT-X | <https://doi.org/10.1016/j.infsof.2023.107390> | <https://github.com/toda-lab/Vbt-X> |

A small number of integration changes standardize inputs and outputs for
automated experiments. The shared method adapters are in
[`Experiments/common/method_execution.py`](../Experiments/common/method_execution.py).
