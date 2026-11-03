from benchmarks.common import BenchmarkBase


class RobomimicTransportMhBenchmark(BenchmarkBase):
    name = "robomimic_transport_mh"
    family = "robomimic"
    task = "Transport"
    dataset_variant = "MH"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "robomimic_transport_mh"
    image_keys_status = "UNVERIFIED"
    expected_dataset_path = "artifacts/robomimic/transport/mh/image.hdf5"


BENCHMARK = RobomimicTransportMhBenchmark()
