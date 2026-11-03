from benchmarks.common import BenchmarkBase


class LiberoHinflowBenchmark(BenchmarkBase):
    name = "libero_hinflow"
    family = "libero"
    task = "HinFlow-LIBERO"
    dataset_variant = "hinflow_original"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "libero_hinflow"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = ""


BENCHMARK = LiberoHinflowBenchmark()
