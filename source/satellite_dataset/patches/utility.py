import user.config as user_config


def passes_resolution_threshold(
    img_max_resolution: float, patch_resolution: float
) -> bool:
    return (
        img_max_resolution / patch_resolution < user_config.PATCH_RESOLUTION_TOLERANCE
    )
