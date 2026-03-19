import os
from dataset import (
    DataLoaderTrain, DataLoaderVal, DataLoaderTest,
    DataLoaderTrainSideBySide, DataLoaderValSideBySide,
)


def get_training_data(rgb_dir, img_options, side_by_side=False):
    assert os.path.exists(rgb_dir), f"Training directory not found: {rgb_dir}"
    if side_by_side:
        return DataLoaderTrainSideBySide(rgb_dir, img_options, None)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir, side_by_side=False):
    assert os.path.exists(rgb_dir), f"Validation directory not found: {rgb_dir}"
    if side_by_side:
        return DataLoaderValSideBySide(rgb_dir, None)
    return DataLoaderVal(rgb_dir, None)


def get_test_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, None)