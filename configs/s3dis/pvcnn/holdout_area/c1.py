"""Configuration of S3DIS holdout area with unity width multiplier."""
from utils.config import configs

configs.submodel.width_multiplier = 1

configs.cloud_branch.width_multiplier = configs.submodel.width_multiplier
configs.classification_head.width_multiplier = configs.submodel.width_multiplier
configs.point_branch.width_multiplier = configs.submodel.width_multiplier
