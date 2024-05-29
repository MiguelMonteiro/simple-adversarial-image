import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.optim import AdamW
from torchvision.models.resnet import ResNet50_Weights, resnet50
from tqdm import tqdm


def print_probs(
    probs: Tensor, class_dict: dict[int, str], title: str = "", tok_k: int = 5
) -> None:
    print(title)
    for index in torch.argsort(probs, descending=True)[0, :tok_k]:
        class_name = class_dict[int(index)]
        prob = float(probs[0, index])
        print(f"\t{class_name}: {prob:.2%}")


def get_imagenet_class_dict() -> dict[int, str]:
    """Returns a dictionary mapping from class indices to class names in ImageNet

    Returns:
        dict[int, str]: The mapping between class indices and class names.
    """
    with open("assets/imagenet_class_index.json", "r") as f:
        class_dict = {int(key): value[1] for key, value in json.load(f).items()}
    return class_dict


def get_resnet50_model() -> tuple[nn.Module, Callable[[Any], Tensor]]:
    """Get the resnet50 pretrained imagenet classifier and the corresponding preprocessing function.

    Returns:
        tuple[nn.Module, Callable[[Any], Tensor]]: A tuple containing the pre-trained model and preprocessing function.
    """
    weights = ResNet50_Weights.DEFAULT
    preprocess = cast(Callable[[Any], Tensor], weights.transforms())
    model = resnet50(weights=weights)
    model.eval()  # set model to eval to disable dropout etc...
    return model, preprocess


def classify(model: nn.Module, image: Tensor) -> Tensor:
    """Returns class probabilities given a pre-processed image and model

    Args:
        model (nn.Module): The model.
        image (Tensor): The pre-processed input image with shape (batch_size, num_channels, width, height).

    Returns:
        Tensor: The ImageNet class probabilities with shape (batch_size, num_classes)
    """
    logits = model(image)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs


def compute_adversarial_noise(
    model: nn.Module, image: Tensor, target_class: int, num_steps: int = 100, c: float = 0.01
) -> Tensor:
    """Computes the "minimal" adversarial noise to add to the image such that the new predicted class will equal target_class

    Args:
        model (nn.Module): The model.
        image (Image): The pre-processed input image with shape (batch_size, num_channels, width, height).
        target_class (int): The target class.
        num_steps (int, optional): The number of optimization steps. Defaults to 100.
        c (float, optional): The constant weighting the cost of the l1-norm of the noise. Defaults to 1.

    Returns:
        Tensor: The computed adversarial noise.
    """

    target = torch.zeros(size=(image.shape[0], 1000))
    target[:, target_class] = 1
    noise = torch.zeros_like(image, requires_grad=True)
    optimizer = AdamW(params=[noise], lr=0.1)
    for _ in tqdm(range(num_steps)):
        optimizer.zero_grad()
        logits = model(image + noise)
        nll = torch.nn.functional.cross_entropy(input=logits, target=target)
        loss = nll + c * torch.norm(noise, p=1)
        loss.backward()
        optimizer.step()
    return noise.detach()


def generate_adversarial_image(
    image_path: Path,
    target_class: int,
    num_steps: int,
    c: float = 0.01,
) -> None:
    if target_class < 0 or target_class > 999:
        raise ValueError(
            f"target class must be a value in the range [0, 999] but was {target_class:d}"
        )
    class_dict = get_imagenet_class_dict()
    model, preprocess = get_resnet50_model()
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    probs = classify(model, image)
    print_probs(probs, class_dict, title="Probabilities for input image:")
    noise = compute_adversarial_noise(model, image, target_class, num_steps=num_steps, c=c)
    probs_adversarial = classify(model, image + noise)
    print_probs(probs_adversarial, class_dict, title="Probabilities for adversarial image:")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=Path, default="assets/elephant.webp")
    parser.add_argument("--target_class", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--c", type=float, default=0.01)
    args, _ = parser.parse_known_args()
    generate_adversarial_image(**args.__dict__)
