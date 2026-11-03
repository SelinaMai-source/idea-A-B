from benchmarks.common import BenchmarkBase


class RobomimicCanMhBenchmark(BenchmarkBase):
    name = "robomimic_can_mh"
    family = "robomimic"
    task = "Can"
    dataset_variant = "MH"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "robomimic_can_mh"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/robomimic/can/mh/image.hdf5"


BENCHMARK = RobomimicCanMhBenchmark()
