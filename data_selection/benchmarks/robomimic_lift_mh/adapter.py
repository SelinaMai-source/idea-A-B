from benchmarks.common import BenchmarkBase


class RobomimicLiftMhBenchmark(BenchmarkBase):
    name = "robomimic_lift_mh"
    family = "robomimic"
    task = "Lift"
    dataset_variant = "MH"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "robomimic_lift_mh"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/robomimic/lift/mh/image.hdf5"


BENCHMARK = RobomimicLiftMhBenchmark()
