from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import v2

from . import conf


datapath = "./data/cifar10/"
pixelmean = [0.49139968, 0.48215841, 0.44653091]
classes = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

transform_normal = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=pixelmean, std=[1.0, 1.0, 1.0]),
    ]
)

device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)


def _visualization_dir() -> Path:
    out_dir = Path(conf.fig_output_filepath) / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _safe_name(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace("\\", "_")


def _normalize_array(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if math.isclose(max_value, min_value):
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def _to_display_image(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().clone()
    for channel, mean in enumerate(pixelmean):
        image[channel] += mean
    image = image.clamp(0.0, 1.0)
    return image.permute(1, 2, 0).numpy()


def _conv_layers(model: nn.Module) -> list[tuple[str, nn.Conv2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]


def _grid_size(count: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    return rows, cols


def _save_grid(
    images: Iterable[np.ndarray],
    title: str,
    output_path: Path,
    cmap: str | None = None,
) -> Path:
    image_list = list(images)
    if not image_list:
        raise ValueError("No images to plot.")

    rows, cols = _grid_size(len(image_list))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes_arr = np.asarray(axes).reshape(rows, cols)

    for index, ax in enumerate(axes_arr.ravel()):
        ax.axis("off")
        if index >= len(image_list):
            continue
        ax.imshow(image_list[index], cmap=cmap)
        ax.set_title(str(index), fontsize=8)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _torch_load(path: Path) -> nn.Module:
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)

    if not isinstance(model, nn.Module):
        raise TypeError(f"{path} does not contain a torch.nn.Module.")
    return model


def find_model_path(model_name: str, layers: int | None = None) -> Path:
    model_dir = Path(conf.model_dump_filepath)
    model_prefix = model_name if layers is None else f"{model_name}_{layers}"
    candidates = [
        path
        for path in model_dir.iterdir()
        if path.is_file() and path.name.startswith(f"{model_prefix}-")
    ]

    if not candidates:
        raise FileNotFoundError(
            f"No saved model matching '{model_prefix}-*' in {model_dir}."
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def load_model(
    model_name: str = "resnet",
    layers: int | None = None,
    model_path: str | Path | None = None,
) -> nn.Module:
    path = Path(model_path) if model_path is not None else find_model_path(model_name, layers)
    model = _torch_load(path)
    model.to(device)
    model.eval()
    return model


def load_cifar10_sample(sample_index: int = 0, train: bool = False) -> tuple[torch.Tensor, int]:
    dataset = datasets.CIFAR10(
        root=datapath,
        train=train,
        download=False,
        transform=transform_normal,
    )
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"sample_index must be in [0, {len(dataset) - 1}].")

    image, label = dataset[sample_index]
    return image, int(label)


def save_input_image(
    image: torch.Tensor,
    label: int,
    output_dir: Path,
    prefix: str,
) -> Path:
    output_path = output_dir / f"{prefix}_input.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(_to_display_image(image))
    ax.axis("off")
    ax.set_title(f"input: {classes[label]}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_conv_filters(
    model: nn.Module,
    output_dir: Path,
    prefix: str,
    max_layers: int = 6,
    max_filters: int = 16,
) -> list[Path]:
    saved_paths: list[Path] = []

    for layer_index, (name, conv) in enumerate(_conv_layers(model)[:max_layers]):
        weights = conv.weight.detach().cpu()
        filters = []
        for filter_weight in weights[:max_filters]:
            if filter_weight.shape[0] == 3:
                filter_image = filter_weight.permute(1, 2, 0).numpy()
                filters.append(_normalize_array(filter_image))
            else:
                filter_image = filter_weight.mean(dim=0).numpy()
                filters.append(_normalize_array(filter_image))

        output_path = output_dir / (
            f"{prefix}_filters_{layer_index:02d}_{_safe_name(name)}.png"
        )
        saved_paths.append(
            _save_grid(
                filters,
                title=f"{prefix} filters: {name}",
                output_path=output_path,
                cmap=None if weights.shape[1] == 3 else "viridis",
            )
        )

    return saved_paths


def capture_feature_maps(
    model: nn.Module,
    image: torch.Tensor,
    max_layers: int = 6,
) -> list[tuple[str, torch.Tensor]]:
    activations: list[tuple[str, torch.Tensor]] = []
    hooks = []

    def make_hook(name: str):
        def hook(_module, _inputs, output):
            activations.append((name, output.detach().cpu()))

        return hook

    for name, conv in _conv_layers(model)[:max_layers]:
        hooks.append(conv.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(image.unsqueeze(0).to(device))

    for hook in hooks:
        hook.remove()

    return activations


def save_feature_maps(
    activations: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    prefix: str,
    max_channels: int = 16,
) -> list[Path]:
    saved_paths: list[Path] = []

    for layer_index, (name, activation) in enumerate(activations):
        feature_map = activation[0]
        images = [
            _normalize_array(channel.numpy())
            for channel in feature_map[:max_channels]
        ]
        output_path = output_dir / (
            f"{prefix}_features_{layer_index:02d}_{_safe_name(name)}.png"
        )
        saved_paths.append(
            _save_grid(
                images,
                title=f"{prefix} feature maps: {name}",
                output_path=output_path,
                cmap="viridis",
            )
        )

    return saved_paths


def save_activation_summary(
    activations: list[tuple[str, torch.Tensor]],
    output_dir: Path,
    prefix: str,
) -> Path:
    output_path = output_dir / f"{prefix}_activation_summary.png"
    labels = [name for name, _ in activations]
    channels = [activation.shape[1] for _, activation in activations]
    heights = [activation.shape[2] for _, activation in activations]
    widths = [activation.shape[3] for _, activation in activations]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.8), 4))
    ax.plot(x, channels, marker="o", label="channels")
    ax.plot(x, heights, marker="s", label="height")
    ax.plot(x, widths, marker="^", label="width")
    ax.set_title(f"{prefix} activation shapes")
    ax.set_xlabel("conv layer")
    ax.set_ylabel("size")
    ax.set_xticks(x)
    ax.set_xticklabels([_safe_name(label) for label in labels], rotation=45, ha="right")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def visualize_model(
    model_name: str = "resnet",
    layers: int | None = None,
    sample_index: int = 0,
    model_path: str | Path | None = None,
    max_layers: int = 6,
    max_filters: int = 16,
    max_channels: int = 16,
) -> list[Path]:
    layers = conf.block_n * 6 + 2 if layers is None and model_path is None else layers
    model = load_model(model_name=model_name, layers=layers, model_path=model_path)
    image, label = load_cifar10_sample(sample_index=sample_index)

    prefix = model_name if layers is None else f"{model_name}_{layers}"
    if model_path is not None:
        prefix = Path(model_path).name

    output_dir = _visualization_dir()
    saved_paths = [save_input_image(image, label, output_dir, prefix)]
    saved_paths.extend(
        save_conv_filters(
            model=model,
            output_dir=output_dir,
            prefix=prefix,
            max_layers=max_layers,
            max_filters=max_filters,
        )
    )
    activations = capture_feature_maps(model=model, image=image, max_layers=max_layers)
    saved_paths.extend(
        save_feature_maps(
            activations=activations,
            output_dir=output_dir,
            prefix=prefix,
            max_channels=max_channels,
        )
    )
    saved_paths.append(
        save_activation_summary(
            activations=activations,
            output_dir=output_dir,
            prefix=prefix,
        )
    )
    return saved_paths


def main(
    model_name: str = "resnet",
    layers: int | None = None,
    sample_index: int = 0,
) -> list[Path]:
    saved_paths = visualize_model(
        model_name=model_name,
        layers=layers,
        sample_index=sample_index,
    )
    for path in saved_paths:
        print(f"saved: {path}")
    return saved_paths


if __name__ == "__main__":
    main()
