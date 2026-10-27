"""The local stock refinement must only use the same saved camera frames."""

import numpy as np
import pytest

from probe_zerodex_saved_current_refine import _bind_views


def test_bind_views_requires_matching_frame_and_mask_hashes():
    view = {"serial": "one", "image_sha256": "image", "mask_sha256": "mask"}
    row = {"serial": "one", "status": "pose_estimated_not_validated",
           "image_sha256": "image", "mask_sha256": "mask",
           "pose_world_m": np.eye(4).tolist()}
    source = {"schema": "precision_insertion_zerodex_saved_foundpose_probe_v1",
              "object_name": "key", "views": [row]}
    assert np.array_equal(_bind_views([view], source, "key")["one"]["pose_world"],
                          np.eye(4))
    row["image_sha256"] = "different"
    with pytest.raises(ValueError, match="frame/mask differs"):
        _bind_views([view], source, "key")
