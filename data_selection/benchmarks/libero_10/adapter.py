class Libero10Benchmark:
    name = "libero_10"
    family = "libero"
    task = "LIBERO-10"
    dataset_variant = "LIBERO-10"
    observation_mode = "image"
    action_convention = "UNVERIFIED"
    forgetting_applicable = False
    protocol_id = "libero_10"
    image_keys_status = "UNVERIFIED"

    def schema(self):
        return {
            "image_keys_status": self.image_keys_status,
            "action_convention": self.action_convention,
            "forgetting_applicable": self.forgetting_applicable,
            "formal_result_without_hdf5_or_zarr": False,
        }


BENCHMARK = Libero10Benchmark()
