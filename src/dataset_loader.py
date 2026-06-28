# src/dataset_loader.py

import random
from datasets import load_dataset


def load_sst2(sample_size=100):

    dataset = load_dataset(
        "nyu-mll/glue",
        "sst2"
    )

    validation = dataset["validation"]

    random.seed(42)

    indices = random.sample(
    range(len(validation)),
    sample_size
)

    subset = validation.select(indices)

    return subset
