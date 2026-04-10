import torch
from torch import optim


class SharedAdam(optim.Adam):
    def __init__(self, params, lr=1e-3, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        # TODO: Initialize optimizer state tensors for every tracked parameter
        for group in self.param_groups:
            for param in group["params"]:
                if param is None:
                    continue

                state = self.state[param]

                # TODO: Create the step counter tensor
                state["step"] = torch.zeros(1)

                # TODO: Create Adam's first-moment buffer
                state["exp_avg"] = torch.zeros_like(param.data)

                # TODO: Create Adam's second-moment buffer
                state["exp_avg_sq"] = torch.zeros_like(param.data)

    def share_memory(self):
        """Move optimizer state to shared memory when supported by the device."""
        # TODO: Move the optimizer state tensors into shared memory
        # Hint: Iterate over the optimizer state attached to each parameter, and only
        # attempt to share tensors that support shared-memory transfer.
        for group in self.param_groups:
            for param in group["params"]:
                state = self.state[param]
                if not state:
                    continue

                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = state.get(key)

                    # TODO: Skip entries that cannot be shared
                    # Hint: Some state entries may be missing or may not expose the method
                    # needed for shared-memory transfer.
                    if value is None or not hasattr(value, "share_memory_"):
                        continue

                    # TODO: Share CPU state tensors across processes
                    # Hint: The shared-parameter A3C setup is designed around CPU shared memory.
                    value.share_memory_()

        return self