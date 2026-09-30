from .conversion import cart2se3, se32cart, se32action
from .robot_config import INIT_STATE, XARM_INIT, ALLEGRO_INIT, LINK6_TO_WRIST
from .path import (
    project_dir, obj_path, urdf_path, robot_configs_path,
    candidate_path, code_path, shared_dir,
    load_candidate, load_openpose_for_candidates, get_object_mesh,
)


def get_scene_image_dict_template(*args, **kwargs):
    """Load ParaDex-backed scene helpers only when camera rendering is used."""
    from .scene import get_scene_image_dict_template as _get_scene_image_dict_template
    return _get_scene_image_dict_template(*args, **kwargs)
