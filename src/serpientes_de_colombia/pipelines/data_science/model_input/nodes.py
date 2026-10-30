import numpy as np
import pandas as pd
from torchvision import  transforms
from torchvision.transforms.functional import to_pil_image
from pathlib import Path
from PIL import ImageDraw


def global_label_encoder(image_dataset):
    image_data= image_dataset.data
    label_map=dict(
        zip(
            image_data[image_dataset.label_column].unique(),
            range(image_data[image_dataset.label_column].nunique())
        )
        )
    return label_map

def labels_previous_check(
    train_image_dataset,
    test_image_dataset,
    validation_image_dataset,
    label_map,
    sample_size=40
):

    reverse_label_map = {v: k for k, v in label_map.items()}

    output = {}

    datasets = {
        "train": train_image_dataset,
        "validation": validation_image_dataset,
        "test": test_image_dataset,
    }

    for split_name, image_dataset in datasets.items():

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
        ])

        target_transform = lambda x: label_map.get(x)

        dataset = image_dataset.with_transforms(
            transform=transform,
            target_transform=target_transform
        )

        random_indexes = np.random.choice(
            range(len(dataset)),
            min(sample_size, len(dataset)),
            replace=False
        )

        for idx in random_indexes:

            x, y, path = dataset[idx]

            # x is already transformed but NOT normalized
            pil_img = to_pil_image(x)

            draw = ImageDraw.Draw(pil_img)

            label_text = (
                f"y={y}\n"
                f"{reverse_label_map[y]}\n"
                f"{Path(path).name}"
            )

            draw.rectangle(
                [(0, 0), (220, 45)],
                fill="white"
            )

            draw.text(
                (5, 5),
                label_text,
                fill="black"
            )

            output[
                f"{split_name}/{reverse_label_map[y]}/{Path(path).stem}.png"
            ] = pil_img

    return output

      
