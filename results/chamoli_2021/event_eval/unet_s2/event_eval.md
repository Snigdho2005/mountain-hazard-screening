# Event evaluation: Chamoli rock-ice avalanche, 7 Feb 2021
run: `runs/chamoli_unet_s2` (unet), 131 zones

## Documented source (within 1 km)
elevation 5288.8 m, slope 39.2 deg, aspect 43.7 deg; usable on both dates 86%; shadow t1 73%, t2 47%, both 47%
classes t1 {'background': 0.001, 'snow_ice': 0.365, 'rock': 0.524, 'soil': 0.096, 'debris': 0.015}; t2 {'background': 0.001, 'snow_ice': 0.541, 'rock': 0.357, 'soil': 0.076, 'debris': 0.025}
changed pixels before zone rules 274; in zones 199

## Ranking of the documented source (energy term is an index (rho*g*area*drop, J per metre of thickness): no thickness scenario declared)

### zones within 500 m: 0 (random expectation for best rank: None)
no zone at the documented source

### zones within 1000 m: 1 (random expectation for best rank: 66.0)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 6 | 4.6% | False | True | True |
| no_energy | 21 | 16.0% | False | False | True |
| no_instability | 21 | 16.0% | False | False | True |
| no_exposure | 2 | 1.5% | True | True | True |
| no_route_(and_corridor_exposure) | 2 | 1.5% | True | True | True |
| baseline_area_only | 37 | 28.2% | False | False | False |
| baseline_confidence_then_area | 35 | 26.7% | False | False | False |
| pipeline_as_run | 21 | 16.0% | False | False | True |

### zones within 2000 m: 1 (random expectation for best rank: 66.0)
| ranking | best rank | percentile | top 5 | top 10 | top 25 |
|---|---|---|---|---|---|
| full | 6 | 4.6% | False | True | True |
| no_energy | 21 | 16.0% | False | False | True |
| no_instability | 21 | 16.0% | False | False | True |
| no_exposure | 2 | 1.5% | True | True | True |
| no_route_(and_corridor_exposure) | 2 | 1.5% | True | True | True |
| baseline_area_only | 37 | 28.2% | False | False | False |
| baseline_confidence_then_area | 35 | 26.7% | False | False | False |
| pipeline_as_run | 21 | 16.0% | False | False | True |

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
