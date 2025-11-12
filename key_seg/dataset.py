"""
Combine and split datasets for training, validation, and testing.
"""

from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from coco import CocoKeyDataset, combine_coco_datasets
import utils

datasets = [
    CocoKeyDataset.from_json_file(
        images_dir="datasets/fur-elise/cropped",
        json_path="datasets/fur-elise/cropped/annotations.json",
        transform=None,
    ),
    CocoKeyDataset.from_json_file(
        images_dir="datasets/clown-balloon/cropped",
        json_path="datasets/clown-balloon/cropped/annotations.json",
        transform=None,
    ),
]


def dataset(
    split: tuple[float, float] = (0.7, 0.15),
    batch_size: int = 32,
    transforms: tuple[transforms.Compose, transforms.Compose, transforms.Compose] = (
        None,
        None,
        None,
    ),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Combine datasets and split into train, validation, and test sets."""
    train_transform, val_transform, test_transform = transforms

    utils.force_remove_directory("datasets/combined_croppped")
    combined_ds = combine_coco_datasets(
        datasets, new_dir="datasets/combined_croppped", transform=None
    )

    train_ds, val_ds, test_ds = random_split(
        combined_ds,
        [
            int(split[0] * len(combined_ds)),
            int(split[1] * len(combined_ds)),
            (
                len(combined_ds)
                - int(split[0] * len(combined_ds))
                - int(split[1] * len(combined_ds))
            ),
        ],
    )
    print("Train size:", len(train_ds))
    print("Validation size:", len(val_ds))
    print("Test size:", len(test_ds))
    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform
    test_ds.dataset.transform = test_transform

    train_loader, val_loader, test_loader = (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False),
    )

    return train_loader, val_loader, test_loader
