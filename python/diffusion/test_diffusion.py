import os

import numpy as np
import testbook


def test_notebook():
    # Set working directory before testing for testing a collection of pytests
    os.chdir(os.path.join(os.path.abspath(os.path.dirname(__file__))))
    with testbook.testbook("./diffusion_on_npu.ipynb", execute=True, timeout=300) as tb:
        train_losses = tb.get("train_losses")
        np.testing.assert_array_less(np.mean(train_losses[-100:]), 0.15)

        # Generated samples are 16x16 RGB pixel art using only palette colors
        samples = np.array(tb.value("samples.tolist()"))
        palette = np.array(tb.value("palette.tolist()"))
        assert samples.shape == (48, 16, 16, 3)
        assert tb.get("pixel_accuracy") > 0.5
        assert set(map(tuple, samples.reshape(-1, 3))) <= set(map(tuple, palette))
