"""
This file contains the implementation of the QLinear layer.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16

import qant_native_computing_toolkit as qant


class QLinear(torch.nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        assert input.dtype == torch.float32, (
            f"QLinear only supports float32 input, but got input with dtype {input.dtype}"
        )

        # Training forward pass
        if self.training:
            # Use the standard linear forward pass from PyTorch for training
            output = torch.nn.functional.linear(input, self.weight, bias=self.bias)
            return output

        # Eval forward pass (no-grad inference)
        else:
            assert input.device.type == "cpu", (
                f"QLinear only supports CPU execution, but got input on device {input.device}"
            )
            # Convert input and weights to bfloat16 numpy arrays
            input_np = input.detach().numpy().astype(bfloat16)
            weights_np = self.weight.data.detach().numpy().astype(bfloat16)

            # Call the linear forward pass on NPU
            output_np = qant.ai.linear_fprop(input_np, weights_np)

            if self.bias is not None:
                bias_np = self.bias.data.detach().numpy().astype(bfloat16)
                return torch.from_numpy((output_np + bias_np).astype(np.float32))
            else:
                return torch.from_numpy(output_np.astype(np.float32))
