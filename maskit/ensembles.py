from collections import deque
from typing import Dict, Optional

from maskit.masks import MaskedCircuit


class Ensemble(object):
    __slots__ = ("dropout", "perturb")

    def __init__(self, dropout: Optional[Dict], *args, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.perturb = True

    def _branch(self, masked_circuit: MaskedCircuit) -> Dict[str, MaskedCircuit]:
        if not self.perturb or self.dropout is None:
            return {"center": masked_circuit}
        branches = {}
        for key in self.dropout:
            branches[key] = MaskedCircuit.execute(masked_circuit, self.dropout[key])
        return branches

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 0
    ):
        """
        The parameter `step_count` defines the number of training steps that are
        executed for each ensemble branch in addition to one training step
        that is done before the branching.
        """
        # first one trainingstep
        params, _cost, _gradient = optimizer.step_cost_and_grad(
            *args, masked_circuit.parameters, masked_circuit=masked_circuit
        )
        masked_circuit.parameters = params

        # then branching
        branches = self._branch(masked_circuit=masked_circuit)
        branch_costs = []
        for branch in branches.values():
            for _ in range(step_count):
                params, _cost, _gradient = optimizer.step_cost_and_grad(
                    *args, branch.parameters, masked_circuit=branch
                )
                branch.parameters = params
            branch_costs.append(args[0](branch.parameters, masked_circuit=branch))

        minimum_index = branch_costs.index(min(branch_costs))
        branch_name = list(branches.keys())[minimum_index]
        selected_branch = branches[branch_name]
        # FIXME: as soon as real (in terms of masked) parameters are used,
        #   no mask has to be applied
        #   until then real gradients must be calculated as gradients also
        #   contain values from dropped gates
        return (selected_branch, branch_name, branch_costs[minimum_index], 0)


class IntervalEnsemble(Ensemble):
    __slots__ = ("_interval", "_counter")

    def __init__(self, dropout: Optional[Dict], interval: int):
        super().__init__(dropout)
        if interval < 1:
            raise ValueError(
                f"Interval must be in a valid range (>= 1), current value {interval}"
            )
        self._interval = interval
        self._counter = 0
        self.perturb = False

    def _check_interval(self):
        if self._counter % self._interval == 0:
            self.perturb = True
            self._counter = 0  # reset the counter
        else:
            self.perturb = False

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 1
    ):
        self._counter += 1
        self._check_interval()
        result = super().step(masked_circuit, optimizer, *args, step_count=step_count)
        return result


class AdaptiveEnsemble(Ensemble):
    __slots__ = ("_cost", "epsilon")

    def __init__(
        self,
        dropout: Optional[Dict[str, Dict]],
        size: int,
        epsilon: float,
    ):
        if size <= 0:
            raise ValueError(f"Size must be bigger than 0 (received {size})")
        super().__init__(dropout)
        self._cost = deque(maxlen=size)
        self.epsilon = epsilon
        self.perturb = False

    def _branch(self, masked_circuit: MaskedCircuit) -> Dict[str, MaskedCircuit]:
        result = super()._branch(masked_circuit)
        self.perturb = False
        return result

    def step(
        self, masked_circuit: MaskedCircuit, optimizer, *args, step_count: int = 1
    ):
        branch, branch_name, branch_cost, gradients = super().step(
            masked_circuit, optimizer, *args, step_count=step_count
        )
        self._cost.append(branch_cost)
        if self._cost.maxlen and len(self._cost) >= self._cost.maxlen:
            if branch_cost > 0.1:  # evaluate current cost
                if (
                    sum(
                        [
                            abs(cost - self._cost[index + 1])
                            for index, cost in enumerate(list(self._cost)[:-1])
                        ]
                    )
                    < self.epsilon
                ):
                    if __debug__:
                        print("======== allowing to perturb =========")
                    self._cost.clear()
                    self.perturb = True
        return (branch, branch_name, branch_cost, gradients)
