# Experiment C: change-detection accuracy, test strip (stratified, area-weighted)

labelled per stratum: {'both': 10, 'unet_only': 4, 'rules_only': 2, 'neither': 11}; skipped/blank: 43
labelled 'change' per stratum: {'both': 10, 'unet_only': 0, 'rules_only': 2, 'neither': 0}
stratum sizes (pixels): {'both': 3487, 'unet_only': 3692, 'rules_only': 70276, 'neither': 601336}


| map | precision | recall | F1 | overall accuracy | est. change area |
|---|---|---|---|---|---|
| c_unet_s0: class change | 0.49 (0.49-0.49) | 0.05 (0.05-0.05) | 0.09 (0.09-0.09) | 0.89 (0.89-0.89) | 10.9% |
| c_unet_s1: class change | 1.00 (1.00-1.00) | 0.51 (0.02-1.00) | 0.68 (0.05-1.00) | 0.95 (0.89-1.00) | 10.9% |
| c_unet_s2: class change | 1.00 (1.00-1.00) | 0.04 (0.02-0.05) | 0.07 (0.05-0.09) | 0.90 (0.89-0.90) | 10.9% |
| rule baseline (classes + CVA) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 1.00 (1.00-1.00) | 10.9% |

Values in brackets: 95% bootstrap intervals. Labels are one analyst's interpretation of 10 m imagery.
