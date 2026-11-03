from core.registry import discover_baselines
from core.seeds import set_seed
from core.status import BLOCKED, OK
from core.trajectories import make_synthetic_trajectory
from core.types import SelectionBudget, SelectionContext


def _pool():
    return [
        make_synthetic_trajectory("a", 3, score=0.2, collected_step=1),
        make_synthetic_trajectory("b", 3, score=0.9, collected_step=5),
        make_synthetic_trajectory("c", 3, score=0.4, collected_step=3),
        make_synthetic_trajectory("d", 5, score=0.1, collected_step=4),
    ]


def _selectors():
    return {item.name: item for item in discover_baselines()}


def test_registry_has_controls_and_papers_and_skips_sources():
    names = set(_selectors())
    assert {"all_data", "random", "fifo", "reservoir", "fixed_score"} <= names
    assert {"cupid", "deminf", "datamil", "demo_score", "strap"} <= names
    assert "hinflow" not in names
    assert "_sources" not in names


def test_budget_reports_episodes_and_transitions():
    rejected = _selectors()["all_data"].select(
        _pool(),
        SelectionBudget(max_episodes=2, max_transitions=4),
        SelectionContext(seed=0),
    )
    assert rejected.status == BLOCKED
    assert rejected.selected_ids == []
    assert rejected.truncated is False
    selected = _selectors()["random"].select(
        _pool(),
        SelectionBudget(max_episodes=2),
        SelectionContext(seed=1),
    )
    assert selected.status == OK
    assert selected.selected_episodes == 2
    assert selected.truncated is False
    widths = {item.trajectory_id: item.n_transitions for item in _pool()}
    assert selected.selected_transitions == sum(widths[item] for item in selected.selected_ids)


def test_random_is_seeded_and_papers_do_not_fall_back():
    selector = _selectors()["random"]
    budget = SelectionBudget(max_episodes=2)
    first = selector.select(_pool(), budget, SelectionContext(seed=7))
    second = selector.select(_pool(), budget, SelectionContext(seed=7))
    third = selector.select(_pool(), budget, SelectionContext(seed=8))
    assert first.selected_ids == second.selected_ids
    assert first.selected_ids != third.selected_ids
    set_seed(7)
    cupid = _selectors()["cupid"].select(_pool(), budget, SelectionContext(seed=7, hindsight_labels={"x": 1}))
    assert cupid.status in {BLOCKED, "SOURCE_UNAVAILABLE", "NOT_IMPLEMENTED"}
    assert cupid.selected_ids == []
    assert "hindsight" not in cupid.information_used
    assert cupid.selected_ids != first.selected_ids or cupid.status != OK


def test_fifo_keeps_newest_and_fixed_score_blocks_without_scores():
    fifo = _selectors()["fifo"].select(_pool(), SelectionBudget(max_episodes=2), SelectionContext(seed=0))
    assert fifo.selected_ids == ["b", "d"]
    blank = [
        make_synthetic_trajectory("a", 2),
        make_synthetic_trajectory("b", 2),
    ]
    blocked = _selectors()["fixed_score"].select(blank, SelectionBudget(max_episodes=1), SelectionContext(seed=0))
    assert blocked.status == BLOCKED
    assert blocked.selected_ids == []
    ranked = _selectors()["fixed_score"].select(
        _pool(),
        SelectionBudget(max_episodes=1),
        SelectionContext(seed=0, score_provenance="unit_test_external_scores"),
    )
    assert ranked.selected_ids == ["b"]
    missing_provenance = _selectors()["fixed_score"].select(
        _pool(),
        SelectionBudget(max_episodes=1),
        SelectionContext(seed=0),
    )
    assert missing_provenance.status == BLOCKED
    assert missing_provenance.selected_ids == []


def test_reservoir_respects_episode_cap_and_seed():
    budget = SelectionBudget(max_episodes=2)
    one = _selectors()["reservoir"].select(_pool(), budget, SelectionContext(seed=3))
    two = _selectors()["reservoir"].select(_pool(), budget, SelectionContext(seed=3))
    assert one.selected_ids == two.selected_ids
    assert one.selected_episodes == 2
    assert one.selected_transitions == sum(3 if item != "d" else 5 for item in one.selected_ids)


def test_strap_is_subtrajectory_and_blocked():
    strap = _selectors()["strap"]
    assert strap.selects == "subtrajectory"
    result = strap.select(_pool(), SelectionBudget(max_episodes=1), SelectionContext(seed=0))
    assert result.selected_ids == []
    assert result.extra_cost["heuristic_fallback"] is False
