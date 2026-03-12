"""
Configuration module replacing pixel3dmm's env_paths.
Provides path configuration for the self-contained pixel3dmm module.
"""
import os

# Module-level path configuration (set during initialization)
_config = None


class Pixel3DMMPaths:
    """Path configuration for pixel3dmm module."""

    def __init__(self,
                 assets_dir,           # pixel3dmm assets (UV coords, masks, head template, etc.)
                 flame_assets_dir,     # FLAME model files (generic_model.pkl, landmark_embedding, etc.)
                 network_ckpt_path=None,    # UV or normals checkpoint path
                 device='cuda:0'):
        self.assets_dir = assets_dir
        self.flame_assets_dir = flame_assets_dir
        self.network_ckpt_path = network_ckpt_path
        self.device = device

        # env_paths compatibility aliases (used by tracker, renderer, losses, etc.)
        self.FLAME_ASSETS = flame_assets_dir
        self.head_template = os.path.join(assets_dir, 'head_template.obj')
        self.head_template_color = os.path.join(assets_dir, 'head_template_color.obj')

        # UV/vertex data
        self.FLAME_UV_COORDS = os.path.join(assets_dir, 'flame_uv_coords.npy')
        self.VERTEX_WEIGHT_MASK = os.path.join(assets_dir, 'flame_vertex_weights.npy')
        self.MIRROR_INDEX = os.path.join(assets_dir, 'flame_mirror_index.npy')
        self.EYE_MASK = os.path.join(assets_dir, 'uv_mask_eyes.png')

        # Valid vertex masks for UV loss
        self.VALID_VERTS = os.path.join(assets_dir, 'uv_valid_verty.npy')
        self.VALID_VERTS_NARROW = os.path.join(assets_dir, 'uv_valid_verty_noEyes.npy')

        # These are only used by the full tracker (not by our initializer)
        self.PREPROCESSED_DATA = '/tmp/pixel3dmm_preprocessed'
        self.TRACKING_OUTPUT = '/tmp/pixel3dmm_tracking'


def init_config(assets_dir, flame_assets_dir, network_ckpt_path=None, device='cuda:0'):
    """Initialize the global path configuration."""
    global _config
    _config = Pixel3DMMPaths(assets_dir, flame_assets_dir, network_ckpt_path, device)
    return _config


def get_config():
    """Get the current path configuration."""
    if _config is None:
        raise RuntimeError("Pixel3DMM paths not initialized. Call init_config() first.")
    return _config


# Proxy attributes for env_paths compatibility
def __getattr__(name):
    config = get_config()
    if hasattr(config, name):
        return getattr(config, name)
    raise AttributeError(f"No path configuration for '{name}'")
