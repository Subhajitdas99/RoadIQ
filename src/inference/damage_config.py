SUPPORTED_DAMAGE_TYPES = {
    "pothole": {
        "display_name": "Pothole",
        "severity_weight": 1.0,
        "color_rgb": (255, 99, 71),
    },
    "crack": {
        "display_name": "Crack",
        "severity_weight": 0.7,
        "color_rgb": (255, 215, 0),
    },
    "waterlogging": {
        "display_name": "Waterlogging",
        "severity_weight": 0.9,
        "color_rgb": (30, 144, 255),
    },
    "road_patch": {
        "display_name": "Road Patch",
        "severity_weight": 0.55,
        "color_rgb": (50, 205, 50),
    },
    "debris": {
        "display_name": "Debris",
        "severity_weight": 0.6,
        "color_rgb": (186, 85, 211),
    },
}

DEFAULT_DAMAGE_CONFIG = {
    "display_name": "Unknown",
    "severity_weight": 0.5,
    "color_rgb": (255, 255, 255),
}
