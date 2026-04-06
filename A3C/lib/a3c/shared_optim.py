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

                state = None  # Replace with your implementation

                # TODO: Create the step counter tensor
                state["step"] = None  # Replace with your implementation

                # TODO: Create Adam's first-moment buffer
                state["exp_avg"] = None  # Replace with your implementation

                # TODO: Create Adam's second-moment buffer
                state["exp_avg_sq"] = None  # Replace with your implementation

    def share_memory(self):
        """Move optimizer state to shared memory when supported by the device."""
        # TODO: Move the optimizer state tensors into shared memory
        # Hint: Iterate over the optimizer state attached to each parameter, and only
        # attempt to share tensors that support shared-memory transfer.
        for group in self.param_groups:
            for param in group["params"]:
                state = None  # Replace with your implementation
                if not state:
                    continue

                for key in ("step", "exp_avg", "exp_avg_sq"):
                    value = None  # Replace with your implementation

                    # TODO: Skip entries that cannot be shared
                    # Hint: Some state entries may be missing or may not expose the method
                    # needed for shared-memory transfer.
                    pass  # Replace with your implementation

                    # TODO: Share CPU state tensors across processes
                    # Hint: The shared-parameter A3C setup is designed around CPU shared memory.
                    pass  # Replace with your implementation

        return self