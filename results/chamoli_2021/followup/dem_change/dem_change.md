# Experiment A: source detection from elevation change

difference: `shean_diff_2m.tif`; covers 67% of the scene
stable terrain: 576851 px, bias -0.31 m (removed), NMAD 3.24 m
lowering threshold: 9.73 m; lowered zones: 552; density [900, 2700] kg/m3

## Top 15 lowered zones (full ranking)
| rank | zone | area (m2) | loss volume (m3) | max lowering (m) | drop (m) | energy median (J) | exposed |
|---|---|---|---|---|---|---|---|
| 1 | 74 | 19,700 | 8.98e+05 | 95.3 | 1468 | 2.32e+13 | 7 |
| 2 | 100 | 35,800 | 7.14e+05 | 33.5 | 555 | 6.84e+12 | 7 |
| 3 | 65 | 19,600 | 2.59e+05 | 17.8 | 1235 | 5.38e+12 | 7 |
| 4 | 70 | 16,800 | 2.14e+05 | 18.4 | 1189 | 4.28e+12 | 7 |
| 5 | 64 | 16,300 | 1.98e+05 | 17.4 | 1121 | 3.72e+12 | 7 |
| 6 | 79 | 8,100 | 2.03e+05 | 46.5 | 1026 | 3.62e+12 | 7 |
| 7 | 69 | 6,900 | 9.92e+04 | 23.9 | 1208 | 2.03e+12 | 7 |
| 8 | 59 | 6,300 | 8.78e+04 | 19.7 | 959 | 1.42e+12 | 7 |
| 9 | 71 | 6,200 | 7.21e+04 | 14.8 | 1124 | 1.35e+12 | 7 |
| 10 | 63 | 6,100 | 7.62e+04 | 17.1 | 1032 | 1.32e+12 | 7 |
| 11 | 126 | 5,600 | 8.83e+04 | 26.6 | 798 | 1.2e+12 | 7 |
| 12 | 67 | 5,000 | 6.25e+04 | 18.0 | 1119 | 1.17e+12 | 7 |
| 13 | 441 | 307,600 | 2.61e+07 | 179.3 | 1626 | 7.49e+14 | 6 |
| 14 | 405 | 291,000 | 1.4e+07 | 182.8 | 1394 | 3.44e+14 | 6 |
| 15 | 203 | 153,900 | 8.45e+06 | 137.5 | 1686 | 2.52e+14 | 6 |

## Documented source: covered by difference True; dz within 1 km {'min_m': -182.81, 'median_m': -0.72}

### lowered zones within 500 m: 3 (random expectation for best rank: 138.2)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 14 | 2.5% | False | False | True |
| no_energy | 14 | 2.5% | False | False | True |
| no_instability | 14 | 2.5% | False | False | True |
| no_exposure | 2 | 0.4% | True | True | True |
| no_route_(and_corridor_exposure) | 2 | 0.4% | True | True | True |
| baseline_area_only | 2 | 0.4% | True | True | True |
| baseline_volume_only | 2 | 0.4% | True | True | True |

### lowered zones within 1000 m: 10 (random expectation for best rank: 50.3)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 14 | 2.5% | False | False | True |
| no_energy | 14 | 2.5% | False | False | True |
| no_instability | 14 | 2.5% | False | False | True |
| no_exposure | 2 | 0.4% | True | True | True |
| no_route_(and_corridor_exposure) | 2 | 0.4% | True | True | True |
| baseline_area_only | 2 | 0.4% | True | True | True |
| baseline_volume_only | 2 | 0.4% | True | True | True |

### lowered zones within 2000 m: 23 (random expectation for best rank: 23.0)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 13 | 2.4% | False | False | True |
| no_energy | 13 | 2.4% | False | False | True |
| no_instability | 13 | 2.4% | False | False | True |
| no_exposure | 1 | 0.2% | True | True | True |
| no_route_(and_corridor_exposure) | 1 | 0.2% | True | True | True |
| baseline_area_only | 1 | 0.2% | True | True | True |
| baseline_volume_only | 1 | 0.2% | True | True | True |

## Source zone vs published values
zone 405 (full rank 14): loss volume 1.4e+07 m3 (95% CI 1.33e+07-1.46e+07); published 2.69e+07 m3; ratio 0.52; published inside CI: False
area 291,000 m2; max lowering 182.8 m; source elevation 4963 m (published ~5500 m); drop 1394 m
mass 2.51e+10 kg (90% 1.38e+10-3.64e+10); energy 3.44e+14 J (90% 1.89e+14-4.98e+14)

| checkpoint | closest (m) | reached | along route (km) | published (km) |
|---|---|---|---|---|
| Ronti Gad | 78.1 | True | 1.97 | None |
| Rishiganga HEP | 28.3 | True | 15.14 | 15 |
| Rishiganga-Dhauliganga confluence | 51.0 | True | 16.7 | None |
| Tapovan HEP | 28.3 | True | 23.05 | 26 |

Energy from measured lowering volume, one declared density range (ice to rock) and the drop from the mass-weighted source elevation to the first valley-floor point on the D8 route. Persistence is not available for DEM zones, so the instability component is constant.
