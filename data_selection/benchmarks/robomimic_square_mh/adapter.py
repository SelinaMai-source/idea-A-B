from benchmarks.common import BenchmarkBase


class RobomimicSquareMhBenchmark(BenchmarkBase):
    name = "robomimic_square_mh"
    family = "robomimic"
    task = "Square"
    dataset_variant = "MH"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "robomimic_square_mh"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/robomimic/square/mh/image.hdf5"


BENCHMARK = RobomimicSquareMhBenchmark()
