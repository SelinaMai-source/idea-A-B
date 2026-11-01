"""arXiv:2609.26651 e-PS (Lin, Ma, Ren, Wei). independent_reimplementation.

Preprint — not published CCF-A. No LLM watermark / 5D / Jester in v0.
"""

from idea_C.baselines.eps.algorithms import CODE_STATUS, run_eps
from idea_C.baselines.eps.dgp import draw_theta, make_eps_instance

__all__ = ["CODE_STATUS", "draw_theta", "make_eps_instance", "run_eps"]
