class LiberoHinflowBenchmark:
    name = "libero_hinflow"
    family = "libero"
    task = "HinFlow-LIBERO"
    dataset_variant = "hinflow_original"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "libero_hinflow"
    image_keys_status = "UNVERIFIED"

    def schema(self):
        return {
            "image_keys_status": self.image_keys_status,
            "action_convention": self.action_convention,
            "forgetting_applicable": self.forgetting_applicable,
            "formal_result_without_hdf5_or_zarr": False,
        }


BENCHMARK = LiberoHinflowBenchmark()
