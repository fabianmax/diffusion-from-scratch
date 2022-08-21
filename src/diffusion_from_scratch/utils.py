from inspect import isfunction
from torchvision import transforms
from torchvision.transforms import Compose


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def img_to_tensor_pipeline(examples):

    # define image transformations (e.g. using torchvision)
    transform_pipeline = Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])

    examples["pixel_values"] = [transform_pipeline(image.convert("L")) for image in examples["image"]]
    del examples["image"]

    return examples


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr
