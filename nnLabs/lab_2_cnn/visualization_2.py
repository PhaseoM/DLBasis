from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def latest_resnet20_path() -> Path:
    return max(Path(conf.model_dump_filepath).glob("resnet_20-*"), key=lambda p: p.stat().st_mtime)


def norm(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def save_input(image: torch.Tensor, label: int, out_dir: Path) -> None:
    img = image.detach().cpu().clone()
    for c, mean in enumerate(pixelmean):
        img[c] += mean
    img = img.clamp(0, 1).permute(1, 2, 0).numpy()

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"input: {classes[label]}")
    plt.tight_layout()
    plt.savefig(out_dir / "00_input.png", dpi=160, bbox_inches="tight")
    plt.close()


def save_channels(name: str, fmap: torch.Tensor, out_dir: Path) -> None:
    fmap = fmap[0].detach().cpu()
    channels = fmap.shape[0]
    cols = 8
    rows = int(np.ceil(channels / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.35, rows * 1.35))
    for i, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.axis("off")
        if i < channels:
            ax.imshow(norm(fmap[i].numpy()), cmap="viridis")
            ax.set_title(str(i), fontsize=7)
    fig.suptitle(f"{name}: {channels} channels")
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}_channels.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mean(name: str, fmap: torch.Tensor, out_dir: Path) -> np.ndarray:
    avg = norm(fmap[0].detach().cpu().mean(dim=0).numpy())
    plt.figure(figsize=(3, 3))
    plt.imshow(avg, cmap="viridis")
    plt.axis("off")
    plt.title(f"{name}: channel mean")
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_mean.png", dpi=160, bbox_inches="tight")
    plt.close()
    return avg


def save_all_means(mean_maps: list[tuple[str, np.ndarray]], out_dir: Path) -> None:
    cols = 5
    rows = int(np.ceil(len(mean_maps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.0))
    for i, ax in enumerate(np.asarray(axes).reshape(-1)):
        ax.axis("off")
        if i < len(mean_maps):
            name, avg = mean_maps[i]
            ax.imshow(avg, cmap="viridis")
            ax.set_title(name, fontsize=8)
    fig.suptitle("feature maps merged by channel mean")
    fig.tight_layout()
    fig.savefig(out_dir / "all_layers_channel_mean.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main(sample_index: int = 0) -> None:
    out_dir = Path(conf.fig_output_filepath) / "visualization_2"
    out_dir.mkdir(parents=True, exist_ok=True)

    test_data = datasets.CIFAR10(
        root=datapath,
        train=False,
        download=False,
        transform=transform_normal,
    )
    image, label = test_data[sample_index]
    image = image.as_subclass(torch.Tensor)
    save_input(image, int(label), out_dir)

    model = torch.load(latest_resnet20_path(), map_location=device, weights_only=False)
    model.to(device).eval()

    layers = [
        ("stem_16", model.nnModel[2]),
        ("block1_1_16", model.nnModel[3]),
        ("block1_2_16", model.nnModel[4]),
        ("block1_3_16", model.nnModel[5]),
        ("block2_1_32", model.nnModel[6]),
        ("block2_2_32", model.nnModel[7]),
        ("block2_3_32", model.nnModel[8]),
        ("block3_1_64", model.nnModel[9]),
        ("block3_2_64", model.nnModel[10]),
        ("block3_3_64", model.nnModel[11]),
    ]

    activations = {}

    def hook(name: str):
        def save_activation(_module, _inputs, output):
            activations[name] = output.detach().cpu()

        return save_activation

    hooks = [layer.register_forward_hook(hook(name)) for name, layer in layers]

    with torch.no_grad():
        model(image.unsqueeze(0).to(device))
    for hook in hooks:
        hook.remove()

    mean_maps = []
    for name, _layer in layers:
        save_channels(name, activations[name], out_dir)
        mean_maps.append((name, save_mean(name, activations[name], out_dir)))
    save_all_means(mean_maps, out_dir)

    print(f"saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
