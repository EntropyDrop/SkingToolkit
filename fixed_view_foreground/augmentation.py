import torch
import torch.nn.functional as F


def _foreground_mean(foreground_rgb, mask):
    weights = mask.float()
    denominator = weights.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    return (foreground_rgb * weights).sum(dim=(2, 3), keepdim=True) / denominator


def _smooth_noise(batch, height, width, device, dtype):
    low_height = max(2, height // 32)
    low_width = max(2, width // 32)
    noise = torch.rand(batch, 3, low_height, low_width, device=device, dtype=dtype)
    return F.interpolate(noise, size=(height, width), mode="bilinear", align_corners=False)


def random_backgrounds(foreground_rgb, foreground_mask):
    """Generate difficult solid, gradient, gray, and near-character backgrounds."""
    batch, _, height, width = foreground_rgb.shape
    device = foreground_rgb.device
    dtype = foreground_rgb.dtype
    mode = torch.randint(0, 5, (batch,), device=device)

    base = torch.rand(batch, 3, 1, 1, device=device, dtype=dtype)
    gray = torch.empty(batch, 1, 1, 1, device=device, dtype=dtype).uniform_(0.55, 0.88)
    gray_rgb = (gray + torch.randn_like(base) * 0.015).clamp(0.0, 1.0)
    near_character = (
        _foreground_mean(foreground_rgb, foreground_mask)
        + torch.randn(batch, 3, 1, 1, device=device, dtype=dtype) * 0.08
    ).clamp(0.0, 1.0)

    base = torch.where((mode == 1).view(-1, 1, 1, 1), gray_rgb, base)
    base = torch.where(
        (mode == 2).view(-1, 1, 1, 1), near_character, base
    )
    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype).view(
        1, 1, height, 1
    )
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype).view(
        1, 1, 1, width
    )
    slope_x = torch.empty(batch, 3, 1, 1, device=device, dtype=dtype).uniform_(
        -0.12, 0.12
    )
    slope_y = torch.empty(batch, 3, 1, 1, device=device, dtype=dtype).uniform_(
        -0.12, 0.12
    )
    gradient = base + slope_x * x + slope_y * y
    textured = gradient + (_smooth_noise(batch, height, width, device, dtype) - 0.5) * 0.08

    gradient_mode = ((mode == 3) | (mode == 1)).view(-1, 1, 1, 1)
    texture_mode = (mode == 4).view(-1, 1, 1, 1)
    background = torch.where(gradient_mode, gradient, base)
    background = torch.where(texture_mode, textured, background)
    return background.clamp(0.0, 1.0)


def composite_random_background(rendered, foreground_target, source_bg_color):
    """Replace a renderer background while retaining antialiased character edges."""
    if rendered.dim() != 4 or rendered.shape[1] != 4:
        raise ValueError(f"Expected RGBA NCHW render, got {tuple(rendered.shape)}.")
    if foreground_target.shape != rendered.shape[:1] + (1,) + rendered.shape[-2:]:
        raise ValueError(
            "foreground_target must have shape "
            f"{rendered.shape[:1] + (1,) + rendered.shape[-2:]}, got "
            f"{tuple(foreground_target.shape)}."
        )
    alpha = rendered[:, 3:4].clamp(0.0, 1.0)
    source_bg = rendered.new_tensor(source_bg_color).view(1, 3, 1, 1)
    if source_bg.max() > 1.0:
        source_bg = source_bg / 255.0
    foreground_rgb = (
        rendered[:, :3] - (1.0 - alpha) * source_bg
    ) / alpha.clamp_min(1e-4)
    foreground_rgb = foreground_rgb.clamp(0.0, 1.0)
    background = random_backgrounds(foreground_rgb, foreground_target)
    composited = alpha * foreground_rgb + (1.0 - alpha) * background
    # Small sensor/compression-style color noise closes the synthetic-to-render gap.
    noise = torch.randn_like(composited) * 0.006
    return (composited + noise).clamp(0.0, 1.0), background
