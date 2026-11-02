import pytest

from policies.diffusion_policy_image.adapter import probe


@pytest.mark.integration
def test_simulator_rollout_requires_cuda():
    report = probe()
    if not report["cuda"]:
        pytest.skip("requires NVIDIA CUDA and a simulator")
    pytest.fail("CUDA is available; a real simulator smoke is not implemented in v0 and must not be marked passed")
