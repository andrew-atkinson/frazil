"""Frazil: shared code for the monthly sea-ice model (the scripts in scripts/ import it).

The variable lists live here so they can be imported without loading PyTorch.
"""
STATES = ["sit", "sic", "sid", "siu", "siv", "snt"]
FORCINGS = ["tus", "huss", "uas", "vas"]
DEGREE = ["pdd_month", "fdd_month", "pdd_year", "fdd_year"]
UNITS = {"sit": "m", "sic": "1", "sid": "1", "siu": "m/s", "siv": "m/s", "snt": "m"}
