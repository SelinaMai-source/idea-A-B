from core.matrix import build_matrix
from core.protocol_validation import ProtocolError, forgetting_cell, validate_comparison_rows
from core.replay_buffer import ReplayBuffer
from core.trajectories import make_synthetic_trajectory
from core.types import SelectionBudget
from evaluation.metrics import success_auc
from online.loop import OnlineLoop


def test_matrix_does_not_mark_ready_from_a_url():
    matrix = build_matrix()
    assert "hinflow" not in {row["method"] for row in matrix["rows"] if row["leaderboard_eligible"]}
    hin = [row for row in matrix["rows"] if row["method"] == "hinflow"]
    assert hin and hin[0]["leaderboard_eligible"] is False
    assert hin[0]["image_dp_compatibility"] == "NOT_APPLICABLE"
    assert all(row["implementation_status"] != "READY" for row in matrix["rows"])
    assert all(job["device"] == "cpu" for job in matrix["runnable_jobs"])
    assert any(item["benchmark"] == "robomimic_lift_mh" and item.get("metric") == "forgetting" for item in matrix["unsupported"])
    names = {row["method"] for row in matrix["rows"]}
    assert "cupid" in names and "random" in names and "hinflow" in names


def test_forgetting_is_na_for_single_task_robomimic():
    assert forgetting_cell("robomimic_lift_mh", None) == "N/A"
    assert forgetting_cell("libero_continual", None) == "NOT_RUN"


def test_mixed_observation_spaces_are_rejected():
    try:
        validate_comparison_rows(
            [
                {"observation_mode": "image", "action_convention": "a", "policy_model": "dp"},
                {"observation_mode": "lowdim", "action_convention": "a", "policy_model": "dp"},
            ]
        )
    except ProtocolError:
        return
    raise AssertionError("mixed observation spaces were accepted")


def test_replay_keeps_failures_and_hindsight_out():
    buffer = ReplayBuffer(allow_hindsight=False)
    assert buffer.add(make_synthetic_trajectory("ok", 2, origin="successful_self_rollout", success=True)) == "accepted"
    assert buffer.add(make_synthetic_trajectory("bad", 2, origin="failed_rollout")) == "held_out_failed"
    assert buffer.add(make_synthetic_trajectory("hind", 2, origin="hindsight_relabel")) == "held_out_hindsight"
    assert buffer.add(make_synthetic_trajectory("exp", 2, origin="external_expert", success=True)) == "accepted"
    assert buffer.expert_query_count == 1
    assert [item.trajectory_id for item in buffer.imitation] == ["ok", "exp"]


def test_synthetic_online_loop_is_labeled_and_matches_the_grid():
    from core.registry import discover_baselines

    selector = next(item for item in discover_baselines() if item.name == "random")
    result = OnlineLoop().run(
        selector,
        seed=0,
        rounds=2,
        episodes_per_round=2,
        transitions_per_episode=4,
        eval_steps=[8, 16],
        budget=SelectionBudget(max_episodes=2),
    )
    assert result["status"] == "SYNTHETIC_INTERFACE_SMOKE"
    assert result["paper_result"] is False
    assert result["training_optimizer_steps"] == 0
    assert result["hindsight_in_imitation"] == 0
    assert result["expert_query_count"] == 1
    assert [step for step, _rate in result["success_curve"]] == [8, 16]
    assert result["final_success_rate"] == result["success_curve"][-1][1]
    assert result["success_auc"] == success_auc(result["success_curve"], [8, 16])
