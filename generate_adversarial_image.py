import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.optim import AdamW, SGD
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


# undoes imagenet normalization so images can be displayed properly
def _undo_normalization(image: Tensor) -> Tensor:
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    for c in range(image.shape[0]):
        image[c] = image[c] * std[c] + mean[c]
    return image


def plot_results(
    image: Tensor,
    noise: Tensor,
    probs_input: Tensor,
    probs_adversarial: Tensor,
    class_dict: dict[int, str],
) -> None:
    """Shows plot/images of the input image, the noise and the adversarial image.

    Args:
        image (Tensor): The input image.
        noise (Tensor): The adversarial noise.
        probs_input (Tensor): The predicted probabilities for the input image.
        probs_adversarial (Tensor): The predicted probabilities for the adversarial image (image + noise)
        class_dict (dict[int, str]): The mapping between class indices and class names.
    """
    images = (image, noise, image + noise)
    titles = ("Input", "Noise", "Input + Noise")
    all_probs = (probs_input, None, probs_adversarial)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, title, img, probs in zip(axes, titles, images, all_probs):
        if probs is not None:
            index = int(torch.argmax(probs[0]))
            prob = float(probs[0, index])
            class_name = class_dict[index]
            ax.text(5, 5, f"{class_name}: {prob:.2%}", bbox={"facecolor": "white", "pad": 10})
        np_img = np.array(_undo_normalization(img[0]).moveaxis(0, -1))
        ax.imshow(np_img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


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


def compute_adversarial_noise_with_lagrangian(
    model: nn.Module, image: Tensor, target_class: int, num_steps: int = 100
) -> Tensor:
    """Computes the "minimal" adversarial noise to add to the image such that the new predicted class will equal target_class

    Args:
        model (nn.Module): The model.
        image (Image): The pre-processed input image with shape (batch_size, num_channels, width, height).
        target_class (int): The target class.
        num_steps (int, optional): The number of optimization steps. Defaults to 100.

    Returns:
        Tensor: The computed adversarial noise.
    """
    epsilon = 0.01
    target = torch.zeros(size=(image.shape[0], 1000))
    target[:, target_class] = 1
    noise = torch.zeros_like(image, requires_grad=True)
    noise_optimizer = AdamW(params=[noise], lr=0.001)
    # Lagrangian multipliers https://www.engraved.blog/how-we-can-make-machine-learning-algorithms-tunable/
    lagrangian_mult = torch.tensor(1.0, requires_grad=True)
    lagrangian_optimizer = SGD(params=[lagrangian_mult], lr=0.1)
    for _ in (pbar := tqdm(range(num_steps))):
        with torch.no_grad():
            lagrangian_mult = lagrangian_mult.clamp_(min=0)

        noise_optimizer.zero_grad()
        lagrangian_optimizer.zero_grad()

        logits = model(image + noise)
        nll = torch.nn.functional.cross_entropy(input=logits, target=target)
        loss = nll - lagrangian_mult * (epsilon - torch.mean(torch.abs(noise)))

        loss.backward()
        noise_optimizer.step()
        lagrangian_mult.grad = -lagrangian_mult.grad  # gradient ascent on multiplier
        lagrangian_optimizer.step()

        pbar.set_description(f"NLL: {float(nll):.4f}   _lambda: {float(lagrangian_mult):.2f}")

    return noise.detach()


def generate_adversarial_image(
    image_path: Path,
    target_class: int,
    num_steps: int,
    c: float,
    use_lagrangian_method: bool,
) -> None:
    if target_class < 0 or target_class > 999:
        raise ValueError(
            f"target class must be a value in the range [0, 999] but was {target_class:d}"
        )
    class_dict = get_imagenet_class_dict()
    model, preprocess = get_resnet50_model()
    image = preprocess(Image.open(image_path)).unsqueeze(0)
    probs_input = classify(model, image)
    print_probs(probs_input, class_dict, title="Probabilities for input image:")
    if use_lagrangian_method:
        noise = compute_adversarial_noise_with_lagrangian(model, image, target_class, num_steps)
    else:
        noise = compute_adversarial_noise(model, image, target_class, num_steps, c=c)
    probs_adversarial = classify(model, image + noise)
    print_probs(probs_adversarial, class_dict, title="Probabilities for adversarial image:")
    plot_results(image, noise, probs_input, probs_adversarial, class_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the adversarial noise necessary to add to an input image such that a resnet classifier misclassifies the image as target_class."
    )
    parser.add_argument(
        "--image_path", type=Path, default="assets/elephant.webp", help="The path to input image."
    )
    parser.add_argument("--target_class", type=int, default=1, help="The target class.")
    parser.add_argument(
        "--num_steps", type=int, default=100, help="The number of optimizer steps."
    )
    parser.add_argument(
        "--c",
        type=float,
        default=0.01,
        help="The regularization constant on the norm of the noise. Is ignored if use_lagrangian_method is True.",
    )
    parser.add_argument(
        "--use_lagrangian_method",
        type=bool,
        default=True,
        help="If true, uses lagrangian multipliers in the computation to avoid having to set the regularization constant on the norm of the noise.",
    )
    args, _ = parser.parse_known_args()
    generate_adversarial_image(**args.__dict__)
