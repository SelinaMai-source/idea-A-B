class RobomimicSquareMhBenchmark:
    name = "robomimic_square_mh"
    family = "robomimic"
    task = "Square"
    dataset_variant = "MH"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "robomimic_square_mh"
    image_keys_status = "UNVERIFIED"

    def schema(self):
        return {
            "image_keys_status": self.image_keys_status,
            "action_convention": self.action_convention,
            "forgetting_applicable": self.forgetting_applicable,
            "formal_result_without_hdf5_or_zarr": False,
        }


BENCHMARK = RobomimicSquareMhBenchmark()
