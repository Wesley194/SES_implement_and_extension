"""Algorithm registry for clean selection via CLI or scripts."""
from functools import partial
from .base import SearchResult
from .best_of_n import best_of_n
from .cem import ses_cem, ses_progressive
from .cma_es import ses_cma
from .saea_gnn import ses_gnn

ALGORITHMS = {
    "BoN": best_of_n,
    "SES(CEM)": ses_cem,
    "SES-CMA": partial(ses_cma, diagonal=False),
    "SES-CMA-diag": partial(ses_cma, diagonal=True),
    "SES-Progressive": ses_progressive,
    "SES-GNN": partial(ses_gnn, bandpass=False),
    "SES-GNN-Bandpass": partial(ses_gnn, bandpass=True),
}


def get_algorithm(name: str):
    if name not in ALGORITHMS:
        raise ValueError(f"Unknown method '{name}'. Available: {list(ALGORITHMS.keys())}")
    return ALGORITHMS[name]