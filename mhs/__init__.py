"""Mountain Hazard Screening (mhs).

Physics-guided screening of mountain hazard source zones from multi-temporal
satellite imagery. Implements the minimum viable pipeline of the concept paper:
input preparation -> surface mapping (one U-Net, six classes) -> change mapping
-> source-zone formation -> physical screening (volume, mass, energy intervals)
-> pathway and exposure screening -> review-state suggestion and run archive.

Outputs are screening products for specialist review. They are not forecasts,
runout predictions or warnings.
"""

__version__ = "0.1.0"

CLASSES = ["background", "water", "snow_ice", "rock", "soil", "debris"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASSES)}
N_CLASSES = len(CLASSES)
IGNORE_LABEL = 255
G = 9.80665  # standard gravity, m s^-2
