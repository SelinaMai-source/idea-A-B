"""Importable API for Idea B v1.1.

Source files live in ``idea_B/methods/v1.1/``. That directory name is not a
valid Python identifier, so this module loads it as the private package
``idea_B._v11_impl`` and re-exports the public names. ``idea_B.methods``
continues to wrap v1.0 and is not modified.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "v1.1"
_PKG_NAME = "idea_B._v11_impl"
_SUBMODULES = (
    "observability",
    "policy_update",
    "feedback_dynamics",
    "identification_test",
    "information_controller",
    "algorithm",
)


def _load_package():
    existing = sys.modules.get(_PKG_NAME)
    if existing is not None and all(
        f"{_PKG_NAME}.{name}" in sys.modules for name in _SUBMODULES
    ):
        return existing

    pkg = types.ModuleType(_PKG_NAME)
    pkg.__path__ = [str(_DIR)]
    pkg.__file__ = str(_DIR / "__init__.py")
    pkg.__package__ = _PKG_NAME
    sys.modules[_PKG_NAME] = pkg

    init_path = _DIR / "__init__.py"
    if init_path.is_file():
        spec = importlib.util.spec_from_file_location(_PKG_NAME, init_path)
        if spec is not None and spec.loader is not None:
            spec.loader.exec_module(pkg)

    for name in _SUBMODULES:
        full = f"{_PKG_NAME}.{name}"
        if full in sys.modules:
            setattr(pkg, name, sys.modules[full])
            continue
        file_path = _DIR / f"{name}.py"
        spec = importlib.util.spec_from_file_location(full, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load v1.1 module {name}")
        module = importlib.util.module_from_spec(spec)
        module.__package__ = _PKG_NAME
        sys.modules[full] = module
        spec.loader.exec_module(module)
        setattr(pkg, name, module)
    return pkg


_pkg = _load_package()
_algorithm = _pkg.algorithm
_controller = _pkg.information_controller
_dynamics = _pkg.feedback_dynamics
_ident = _pkg.identification_test
_obs = _pkg.observability
_policy = _pkg.policy_update

METHOD_FACTORIES = _algorithm.METHOD_FACTORIES
classify_collapse = _dynamics.classify_collapse
run_closed_loop = _algorithm.run_closed_loop
load_config = _algorithm.load_config
allocate_audit = _controller.allocate_audit
allocate_uniform = _controller.allocate_uniform
allocate_equal_expected_count = _controller.allocate_equal_expected_count
allocate_proportional_frequency = _controller.allocate_proportional_frequency
allocate_e_optimal = _controller.allocate_e_optimal
allocate_d_optimal = _controller.allocate_d_optimal
allocate_v11c = _controller.allocate_v11c
expected_skill_audit_counts = _controller.expected_skill_audit_counts
match_mean_budget = _controller.match_mean_budget
make_controller_state = _controller.make_controller_state
accumulate_conditional_gramian = _obs.accumulate_conditional_gramian
accumulate_observed_gramian = _obs.accumulate_observed_gramian
accumulate_candidates_gramian = _obs.accumulate_candidates_gramian
all_candidates_gramian = _obs.all_candidates_gramian
conditional_design_gramian = _obs.conditional_design_gramian
observed_design_gramian = _obs.observed_design_gramian
condition_number = _obs.condition_number
eigenspectrum = _obs.eigenspectrum
lambda_min = _obs.lambda_min
numerical_rank = _obs.numerical_rank
is_psd = _obs.is_psd
skill_information = _obs.skill_information
SlidingWindowGramian = _obs.SlidingWindowGramian
information_bundle = _obs.information_bundle
ridge_update = _policy.ridge_update
sgd_update = _policy.sgd_update
accumulate_xy = _policy.accumulate_xy
observation_probability = _dynamics.observation_probability
excess_mse = _dynamics.excess_mse
worst_skill_mse = _dynamics.worst_skill_mse
population_weighted_mse = _dynamics.population_weighted_mse
parameter_error = _dynamics.parameter_error
recovery_time = _dynamics.recovery_time
cusum_detection_delay = _dynamics.cusum_detection_delay
NOISE_FLOOR = _dynamics.NOISE_FLOOR
two_latent_states = _ident.two_latent_states
observed_log_likelihood = _ident.observed_log_likelihood
fisher_information = _ident.fisher_information
likelihoods_equal = _ident.likelihoods_equal
minimax_identification_lower_bound = _ident.minimax_identification_lower_bound
audit_restores_identifiability = _ident.audit_restores_identifiability
observational_equivalence_report = _ident.observational_equivalence_report
structural_unrecoverable = _ident.structural_unrecoverable
coordinate_in_observed_kernel = _ident.coordinate_in_observed_kernel

__all__ = [
    "METHOD_FACTORIES",
    "NOISE_FLOOR",
    "SlidingWindowGramian",
    "accumulate_candidates_gramian",
    "accumulate_conditional_gramian",
    "accumulate_observed_gramian",
    "accumulate_xy",
    "all_candidates_gramian",
    "allocate_audit",
    "allocate_d_optimal",
    "allocate_e_optimal",
    "allocate_equal_expected_count",
    "allocate_proportional_frequency",
    "allocate_uniform",
    "allocate_v11c",
    "audit_restores_identifiability",
    "classify_collapse",
    "condition_number",
    "conditional_design_gramian",
    "coordinate_in_observed_kernel",
    "cusum_detection_delay",
    "eigenspectrum",
    "excess_mse",
    "expected_skill_audit_counts",
    "fisher_information",
    "information_bundle",
    "is_psd",
    "lambda_min",
    "likelihoods_equal",
    "load_config",
    "make_controller_state",
    "match_mean_budget",
    "minimax_identification_lower_bound",
    "numerical_rank",
    "observation_probability",
    "observational_equivalence_report",
    "observed_design_gramian",
    "observed_log_likelihood",
    "parameter_error",
    "population_weighted_mse",
    "recovery_time",
    "ridge_update",
    "run_closed_loop",
    "sgd_update",
    "skill_information",
    "structural_unrecoverable",
    "two_latent_states",
    "worst_skill_mse",
]
