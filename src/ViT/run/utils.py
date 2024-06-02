"""
Module containing utility functions for data transformations.
"""

from torchvision import transforms


def get_transforms(transform):
    """
    Composes a list of transformations based on the given configuration.

    Args:
        transform (dict): A dictionary containing transformation details.
            - type (str): Type of transformation, either "ToTensor" or "Normalize".
            - mean (tuple): Mean values for normalization (optional, required for Normalize transformation).
            - std (tuple): Standard deviation values for normalization (optional, required for Normalize transformation).

    Returns:
        transforms.Compose: A composed transformation.

    Example:
        >>> transform = {
        ...     "type": "ToTensor"
        ... }
        >>> composed_transform = get_transforms(transform)
    """
    transform_list = []
    if transform.type == "ToTensor":
        transform_list.append(transforms.ToTensor())
    elif transform.type == "Normalize":
        mean = transform.mean
        std = transform.std
        transform_list.append(transforms.Normalize(mean=mean, std=std))
    return transforms.Compose(transform_list)
