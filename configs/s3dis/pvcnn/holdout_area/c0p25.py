"""Configuration of S3DIS holdout area with width multiplier 0.25."""
from utils.config import configs

configs.submodel.width_multiplier = 0.25

configs.cloud_branch.width_multiplier = configs.submodel.width_multiplier
configs.classification_head.width_multiplier = configs.submodel.width_multiplier
configs.point_branch.width_multiplier = configs.submodel.width_multiplier
