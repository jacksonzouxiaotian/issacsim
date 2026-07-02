# Narrow-Passage Evaluation Tables

## Width, Recovery, and Generalization Summary

| controller | scenario | zero_memory | width | clean_SR | raw_SR | collision | wedge | reject | time | min_clearance | osc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| checkpoint | nominal | 0 | 0.75 | 0.953 | nan | 0.031 | 0.000 | 0.000 | 6.774 | 0.308 | 58.41 |
| checkpoint | nominal | 0 | 0.85 | 1.000 | nan | 0.000 | 0.000 | 0.000 | 7.290 | 0.375 | 66.54 |
| checkpoint | nominal | 0 | 0.95 | 1.000 | nan | 0.000 | 0.000 | 0.000 | 6.595 | 0.408 | 51.06 |
| checkpoint | left_wall | 0 | 0.85 | 0.250 | 0.541 | 0.665 | 0.000 | 0.000 | 4.649 | 0.177 | 63.10 |
| checkpoint | left_wall | 1 | 0.85 | 0.000 | nan | 0.469 | 0.000 | 0.000 | 8.147 | 0.181 | 122.06 |
| checkpoint | yaw_left | 0 | 0.85 | 0.208 | 0.245 | 0.464 | 0.003 | 0.000 | 7.416 | 0.339 | 71.71 |

## Delta-D Calibration Inputs

| controller | scenario | delta_D | success_prob | wedge_prob | reject_prob | calibration_error |
| --- | --- | --- | --- | --- | --- | --- |
| checkpoint | nominal | 0.030000000000000027 | 0.953 | 0.000 | 0.000 | 0.047 |
| checkpoint | nominal | 0.13 | 1.000 | 0.000 | 0.000 | 0.000 |
| checkpoint | nominal | 0.22999999999999998 | 1.000 | 0.000 | 0.000 | 0.000 |
| checkpoint | left_wall | 0.13 | 0.239 | 0.000 | 0.000 | 0.761 |
| checkpoint | yaw_left | 0.13 | 0.208 | 0.003 | 0.000 | 0.792 |
