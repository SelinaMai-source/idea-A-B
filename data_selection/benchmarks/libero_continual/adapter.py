from benchmarks.common import BenchmarkBase


class LiberoContinualBenchmark(BenchmarkBase):
    name = "libero_continual"
    family = "libero"
    task = "task_incremental"
    dataset_variant = "constructed"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = True
    protocol_id = "libero_continual"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/libero/continual"


BENCHMARK = LiberoContinualBenchmark()
