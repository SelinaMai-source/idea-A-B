from benchmarks.common import BenchmarkBase


class Libero10Benchmark(BenchmarkBase):
    name = "libero_10"
    family = "libero"
    task = "LIBERO-10"
    dataset_variant = "LIBERO-10"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "libero_10"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/libero/libero_10"


BENCHMARK = Libero10Benchmark()
