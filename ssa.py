import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
from met.gillespie import sim

MODEL = "RNATurnover"

if MODEL == "AutoReg":
    from models.autoreg import Model, args
elif MODEL == "BirthDeath":
    from models.birth_death import Model, args
elif MODEL == "GeneExpr":
    from models.gene_expr import Model, args
elif MODEL == "LotkaVolterra":
    from models.lotka_volterra import Model, args
elif MODEL == "ToggleSwitch":
    from models.toggle_switch import Model, args
elif MODEL == "RNATurnover":
    from models.rna_turnover import Model, args
elif MODEL == "Cascade":
    from models.cascade import Model, args
else:
    raise KeyError("Model not included.")

cme_model = Model(**vars(args))

if __name__ == "__main__":
    # biocircuits: should justify if the current is the main proces

    sim(cme_model, args)
