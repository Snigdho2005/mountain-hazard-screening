# Event evaluation: Chamoli rock-ice avalanche, 7 Feb 2021
run: `runs/chamoli_rules` (rule_baseline), 434 zones

## Documented source (within 1 km)
elevation 5288.8 m, slope 39.2 deg, aspect 43.7 deg; usable on both dates 86%; shadow t1 73%, t2 47%, both 47%
classes t1 {'background': 0.007, 'water': 0.003, 'snow_ice': 0.416, 'rock': 0.393, 'soil': 0.012, 'debris': 0.169}; t2 {'background': 0.004, 'water': 0.002, 'snow_ice': 0.558, 'rock': 0.325, 'soil': 0.009, 'debris': 0.101}
changed pixels before zone rules 763; in zones 150

## Ranking of the documented source (energy term is an index (rho*g*area*drop, J per metre of thickness): no thickness scenario declared)

### zones within 500 m: 0 (random expectation for best rank: None)
no zone at the documented source

### zones within 1000 m: 1 (random expectation for best rank: 217.5)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 12 | 2.8% | False | False | True |
| no_energy | 28 | 6.5% | False | False | False |
| no_instability | 95 | 21.9% | False | False | False |
| no_exposure | 7 | 1.6% | False | True | True |
| no_route_(and_corridor_exposure) | 7 | 1.6% | False | True | True |
| baseline_area_only | 142 | 32.7% | False | False | False |
| baseline_confidence_then_area | 142 | 32.7% | False | False | False |
| pipeline_as_run | 29 | 6.7% | False | False | False |

### zones within 2000 m: 4 (random expectation for best rank: 87.0)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 12 | 2.8% | False | False | True |
| no_energy | 21 | 4.8% | False | False | True |
| no_instability | 95 | 21.9% | False | False | False |
| no_exposure | 7 | 1.6% | False | True | True |
| no_route_(and_corridor_exposure) | 7 | 1.6% | False | True | True |
| baseline_area_only | 58 | 13.4% | False | False | False |
| baseline_confidence_then_area | 58 | 13.4% | False | False | False |
| pipeline_as_run | 22 | 5.1% | False | False | True |

## pathway from documented source
length 30.65 km; reached 4/4 checkpoints (in order: True); flags ['route_crosses_34m_filled_depression_check_lake_dam_or_dem_artefact']
| checkpoint | closest (m) | reached | along route (km) | published (km) |
|---|---|---|---|---|
| Ronti Gad | 78.1 | True | 2.41 | None |
| Rishiganga HEP | 28.3 | True | 15.58 | 15 |
| Rishiganga-Dhauliganga confluence | 51.0 | True | 17.14 | None |
| Tapovan HEP | 28.3 | True | 23.49 | 26 |

## pathway from best matched zone
length 31.21 km; reached 3/4 checkpoints (in order: True); flags ['route_crosses_34m_filled_depression_check_lake_dam_or_dem_artefact']
| checkpoint | closest (m) | reached | along route (km) | published (km) |
|---|---|---|---|---|
| Ronti Gad | 1282.3 | False | 1.44 | None |
| Rishiganga HEP | 28.3 | True | 16.14 | 15 |
| Rishiganga-Dhauliganga confluence | 51.0 | True | 17.7 | None |
| Tapovan HEP | 28.3 | True | 24.05 | 26 |

## Exposure along the route from the documented source
inventory 45 assets; 6 in corridor; documented damaged sites in corridor 3/3, matched by inventory 1/3
- Reni Bridge (Destroyed) (bridge) at 16.6 km
- Reni (settlement) at 16.9 km
- Joshimath-Malari Road (bridge) at 18.0 km
- unnamed bridge (bridge) at 24.3 km
- Garghaniya (settlement) at 25.0 km
- unnamed bridge (bridge) at 25.2 km
