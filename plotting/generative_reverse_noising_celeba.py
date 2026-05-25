from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter


plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


CELEBA_DIR = Path("/home/zongchen/datasets/celeba/img_align_celeba/img_align_celeba")
IMAGE_PATH = CELEBA_DIR / "000002.jpg"
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATH = SCRIPT_DIR / "figures/generative_reverse_noising_celeba.pdf"


def load_target(path, size=112):
    image = Image.open(path).convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=float) / 255.0


def reverse_noising_frames(target, seed=8):
    rng = np.random.default_rng(seed)
    noise = rng.random(target.shape)
    frames = []
    strengths = np.array([0.0, 0.28, 0.52, 0.78, 1.0])

    target_image = Image.fromarray((target * 255).astype(np.uint8))
    for strength in strengths:
        blur_radius = (1.0 - strength) * 9.0
        blurred = np.asarray(target_image.filter(ImageFilter.GaussianBlur(blur_radius)), dtype=float) / 255.0

        coarse_noise = rng.normal(0.0, 1.0, (14, 14, 3))
        coarse_noise = Image.fromarray(((coarse_noise - coarse_noise.min()) / np.ptp(coarse_noise) * 255).astype(np.uint8))
        coarse_noise = coarse_noise.resize(target.shape[:2], Image.Resampling.BICUBIC)
        coarse_noise = np.asarray(coarse_noise, dtype=float) / 255.0

        fine_noise = rng.normal(0.0, 0.20 * (1.0 - strength) ** 1.15, target.shape)
        colour_noise = 0.55 * noise + 0.45 * coarse_noise
        frame = strength * blurred + (1.0 - strength) * colour_noise + fine_noise
        frames.append(np.clip(frame, 0.0, 1.0))

    frames[0] = noise
    frames[-1] = target
    return frames


target = load_target(IMAGE_PATH)
frames = reverse_noising_frames(target)

fig, axes = plt.subplots(1, len(frames), figsize=(8.4, 1.8))
fig.subplots_adjust(left=0.01, right=0.99, bottom=0.03, top=0.97, wspace=0.08)
for idx, (ax, frame) in enumerate(zip(axes, frames)):
    ax.imshow(frame)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)

for idx in range(len(axes) - 1):
    left = axes[idx].get_position()
    right = axes[idx + 1].get_position()
    x0 = left.x1 + 0.006
    x1 = right.x0 - 0.006
    y = (left.y0 + left.y1) / 2
    axes[idx].annotate(
        "",
        xy=(x1, y),
        xytext=(x0, y),
        xycoords=fig.transFigure,
        textcoords=fig.transFigure,
        arrowprops=dict(arrowstyle="->", color="#4f6f8f", lw=1.2),
        annotation_clip=False,
    )

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
