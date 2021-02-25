import random
import pennylane.numpy as np
from enum import Enum

random.seed(1337)


class PerturbationAxis(Enum):
    #: Perturbation affects whole wires
    WIRES = 0
    #: Perturbation affects whole layers
    LAYERS = 1
    #: Perturbation affects random locations
    RANDOM = 2


class PerturbationMode(Enum):
    #: Adding new holes to the mask
    ADD = 0
    #: Removing holes from the mask
    REMOVE = 1
    #: Invert current state of the mask
    INVERT = 2


class MaskedParameters(object):
    """
    TODO: check if np.array.size is wires or layers
    TODO: currently only works for 2d arrays
    TODO: interpretation of wires and layers is not strict and depends on user
        interpretation how the different parameters are mapped to the circuit
    """
    def __init__(self, params, perturbation_axis: PerturbationAxis = PerturbationAxis.RANDOM):
        self._params = params
        self._mask = np.zeros_like(params, dtype=bool, requires_grad=False)
        self.perturbation_axis = perturbation_axis

    @property
    def params(self):
        return self._params

    @property
    def mask(self):
        return self._mask

    @params.setter
    def params(self, values):
        self._params = values

    def copy(self) -> 'MaskedParameters':
        clone = object.__new__(MaskedParameters)
        clone._params = self._params.copy()
        clone._mask = self._mask.copy()
        clone.perturbation_axis = self.perturbation_axis
        return clone

    def perturb(self, amount: int = None, mode: PerturbationMode = PerturbationMode.INVERT):
        assert amount >= 0, "Negative values are not supported, plese use PerturbationMode.REMOVE"
        if mode == PerturbationMode.ADD:
            raise NotImplementedError(f"The mode {mode} is not yet implemented")
        if self.perturbation_axis == PerturbationAxis.WIRES:
            self._perturb_wires(amount, mode)
        elif self.perturbation_axis == PerturbationAxis.LAYERS:
            self._perturb_layers(amount, mode)
        elif self.perturbation_axis == PerturbationAxis.RANDOM:
            self._perturb_random(amount, mode)
        else:
            raise ValueError(f"The perturbation {self.perturbation_axis} is not supported")

    def _perturb_wires(self, amount: int = None, mode: PerturbationMode = PerturbationMode.INVERT):
        wire_count = self._params.shape[1]
        count = abs(amount) if amount is not None else random.randrange(0, wire_count)
        if mode == PerturbationMode.REMOVE:
            indices = [index for index, value in enumerate(self._mask[:, 0]) if value]
            if len(indices) == 0:
                return
        else:
            indices = np.arange(wire_count)
        indices = np.random.choice(indices, min(count, len(indices)), replace=False)
        self._mask[indices] = ~self._mask[indices]

    def _perturb_layers(self, amount: int = None, mode: PerturbationMode = PerturbationMode.INVERT):
        layer_count = self._params.shape[0]
        count = abs(amount) if amount is not None else random.randrange(0, layer_count)
        if mode == PerturbationMode.REMOVE:
            indices = [index for index, value in enumerate(self._mask[0]) if value]
            if len(indices) == 0:
                return
        else:
            indices = np.arange(layer_count)
        layer_indices = [slice(None, None, None), np.random.choice(indices, min(count, len(indices)), replace=False)]
        self._mask[layer_indices] = ~self._mask[layer_indices]

    def _perturb_random(self, amount: int = None, mode: PerturbationMode = PerturbationMode.INVERT):
        count = abs(amount) if amount is not None else random.randrange(0, self._params.size)
        if mode == PerturbationMode.REMOVE:
            indices = np.argwhere(self._mask)
            if len(indices) == 0:
                return
            random_indices = tuple(zip(*indices[np.random.choice(len(indices), min(count, len(indices)), replace=False)]))
        else:
            indices = np.arange(self._params.size)
            selection = np.random.choice(indices, min(count, len(indices)), replace=False)
            random_indices = np.unravel_index(selection, self._mask.shape)
        self._mask[random_indices] = ~self._mask[random_indices]

    def apply_mask(self, params):
        params = np.array(params[0])
        return params[~self._mask]


if __name__ == "__main__":
    parameter = MaskedParameters(np.array(([21, 22, 23], [11, 22, 33], [43, 77, 89])))