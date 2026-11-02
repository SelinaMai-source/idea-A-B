class LiberoContinualBenchmark:
    name = "libero_continual"
    family = "libero"
    task = "task_incremental"
    dataset_variant = "constructed"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = True
    protocol_id = "libero_continual"
    image_keys_status = "UNVERIFIED"

    def schema(self):
        return {
            "image_keys_status": self.image_keys_status,
            "action_convention": self.action_convention,
            "forgetting_applicable": self.forgetting_applicable,
            "formal_result_without_hdf5_or_zarr": False,
        }


BENCHMARK = LiberoContinualBenchmark()
