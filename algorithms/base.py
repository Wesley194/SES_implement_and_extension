"""Shared data structures for inference-time search algorithms."""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class SearchResult:
    best_x0: np.ndarray
    best_score: float
    history: List[Tuple[int, float]] = field(default_factory=list)
    final_x0: Optional[np.ndarray] = None  # e.g., SES Phase-3 sampled u*