from benchmarks.common import BenchmarkBase


class Libero90Benchmark(BenchmarkBase):
    name = "libero_90"
    family = "libero"
    task = "LIBERO-90"
    dataset_variant = "LIBERO-90"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "libero_90"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/libero/libero_90"


BENCHMARK = Libero90Benchmark()
