import numpy as np
from axengine import _C


class InferenceSession:
    def __init__(self, handle) -> None:
        """
        InferenceSession Collection.
        """
        self._handle = handle
        self._init_device()

    def _init_device(self) -> None:
        success = self._handle.init_device()
        if not success:
            raise SystemError("Err... Something wrong while initializing the AX System.")

    @classmethod
    def load_from_model(cls, model_path: str) -> "InferenceSession":
        """
        Load model graph to InferenceSession.

        Args:
            model_path (string): Path to model
        """
        _handle = _C.Runner()
        sess = cls(_handle)
        success = sess._handle.load_model(model_path)
        if not success:
            raise BufferError("Err... Something wrong while loading the model.")
        return sess

    def get_cmm_usage(self):
        return self._handle.get_cmm_usage()

    def feed_input_to_index(self, input_datum: np.ndarray, input_index: int):
        success = self._handle.feed_input_to_index(input_datum, input_index)
        if not success:
            raise BufferError(f"Err... Something wrong while reading the {input_index}th input.")

    def get_output_from_index(self, output_index: int):
        return self._handle.get_output_from_index(output_index)

    def get_inputs(self):
        return self._handle.get_input_names()

    def get_outputs(self):
        return self._handle.get_output_names()

    def get_output_shapes(self):
        return self._handle.get_output_shapes()

    def run(self, input_feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        """
        Returns:
            list[np.ndarray]: Output of the models.
        """
        for i, input_name in enumerate(self.get_inputs()):
            input_datum = input_feed[input_name].flatten()
            self.feed_input_to_index(input_datum, i)

        # Forward
        self._handle.forward()
        # Get outputs
        output_data = {}
        output_shapes = self.get_output_shapes()
        for i, output_name in enumerate(self.get_outputs()):
            output_data[output_name] = self.get_output_from_index(i).reshape(*output_shapes[i])
        return output_data
