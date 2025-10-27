"""
Minimal, readable 3D Gaussian Splatting (3DGS) in pure PyTorch
- Single-file demo that renders a set of 3D Gaussians to a 2D image.
- Differentiable end-to-end (you can optimize positions/colors against a target).
- CPU/MPS/CUDA compatible (defaults to MPS on Apple Silicon when available).

This is intentionally tiny and slow (O(N * average_footprint)) and uses no acceleration structures.
It's great for learning and toy experiments, not for large scenes.

Usage
-----
python 3dgs.py

What you'll see
---------------
A 256x256 image with a few colored 3D Gaussian blobs.

Notes
-----
• We use isotropic 3D covariance (single sigma per Gaussian). Extending to full anisotropic Σ is noted.
• 2D screen-space covariance is derived by linearizing the camera projection.
• Front-to-back alpha compositing with per-splat soft opacity.
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def look_at(eye: torch.Tensor, at: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Build a right-handed view (camera-to-world inverse) matrix [R|t].
    Returns 4x4 world->camera matrix (i.e., view matrix).
    """
    f = (at - eye)
    f = f / (f.norm() + 1e-9)
    u = up / (up.norm() + 1e-9)
    s = torch.cross(f, u, dim=0)
    s = s / (s.norm() + 1e-9)
    u = torch.cross(s, f, dim=0)

    R = torch.stack([s, u, f], dim=0)  # 3x3
    t = -R @ eye.view(3, 1)             # 3x1
    M = torch.eye(4, device=eye.device)
    M[:3, :3] = R
    M[:3, 3:] = t
    return M


def project_points(Xc: torch.Tensor, K: torch.Tensor):
    """Project camera-space points Xc (N,3) with intrinsics K (3,3).
    Returns pixel coords (N,2) and depths z (N,).
    """
    x = Xc[:, 0] / (Xc[:, 2] + 1e-9)
    y = Xc[:, 1] / (Xc[:, 2] + 1e-9)
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u = fx * x + cx
    v = fy * y + cy
    return torch.stack([u, v], dim=-1), Xc[:, 2]


def jacobian_proj(Xc: torch.Tensor, K: torch.Tensor):
    """Compute 2x3 Jacobian of pinhole projection at each camera-space point.
    For point (X,Y,Z):
      u = fx * X/Z + cx, v = fy * Y/Z + cy
    du/dX = fx/Z,  du/dY = 0,       du/dZ = -fx*X/Z^2
    dv/dX = 0,     dv/dY = fy/Z,    dv/dZ = -fy*Y/Z^2
    Returns J of shape (N, 2, 3).
    """
    X, Y, Z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    fx, fy = K[0, 0], K[1, 1]
    invZ = 1.0 / (Z + 1e-9)
    invZ2 = invZ * invZ
    J = torch.zeros((Xc.shape[0], 2, 3), device=Xc.device)
    J[:, 0, 0] = fx * invZ
    J[:, 0, 2] = -fx * X * invZ2
    J[:, 1, 1] = fy * invZ
    J[:, 1, 2] = -fy * Y * invZ2
    return J


def world_to_camera(Xw: torch.Tensor, view: torch.Tensor):
    Xw_h = torch.cat([Xw, torch.ones_like(Xw[:, :1])], dim=-1)  # (N,4)
    Xc_h = (view @ Xw_h.T).T                                   # (N,4)
    return Xc_h[:, :3]

# ──────────────────────────────────────────────────────────────────────────────
# Core 3DGS renderer (naive splatting)
# ──────────────────────────────────────────────────────────────────────────────

def render_gaussians(
    means_world: torch.Tensor,      # (N,3)
    sigmas_world: torch.Tensor,     # (N,) isotropic std-dev in meters
    colors: torch.Tensor,           # (N,3) in [0,1]
    opacities: torch.Tensor,        # (N,) in [0,1]
    view: torch.Tensor,             # (4,4) world->camera
    K: torch.Tensor,                # (3,3) intrinsics
    H: int,
    W: int,
    coverage: float = 3.0,          # bbox = ±coverage * sigma in screen space
    bg: torch.Tensor | None = None,
):
    device = means_world.device
    if bg is None:
        bg = torch.ones(1, 1, 3, device=device)  # white background

    N = means_world.shape[0]

    # Transform to camera space and project
    Xc = world_to_camera(means_world, view)            # (N,3)
    uv, z = project_points(Xc, K)                      # (N,2), (N,)

    # Cull points behind the camera or far away
    valid = (z > 1e-3)
    if valid.sum() == 0:
        return bg.expand(H, W, 3).clone()

    uv = uv[valid]
    z = z[valid]
    Xc = Xc[valid]
    sigmas_world = sigmas_world[valid]
    colors = colors[valid]
    opacities = opacities[valid]

    # 2D covariance via linearization: Σ_img = J Σ_3D J^T, with Σ_3D = σ^2 I
    J = jacobian_proj(Xc, K)                 # (n,2,3)
    sigma2 = (sigmas_world ** 2).view(-1, 1, 1)
    Sigma_img = torch.matmul(J, torch.matmul(sigma2 * torch.eye(3, device=device), J.transpose(1, 2)))  # (n,2,2)

    # Ensure positive-definiteness and extract per-axis std (approximate footprint)
    eps = 1e-6
    # Compute axis-aligned std by taking sqrt of diag; for full ellipse, you'd use inverse Sigma for Mahalanobis.
    std_u = torch.sqrt(Sigma_img[:, 0, 0].clamp_min(eps))
    std_v = torch.sqrt(Sigma_img[:, 1, 1].clamp_min(eps))

    # Depth sort: with +Z forward, nearer points have larger z; composite front-to-back (near first)
    sort_idx = torch.argsort(z, descending=True)
    uv = uv[sort_idx]
    std_u = std_u[sort_idx]
    std_v = std_v[sort_idx]
    colors = colors[sort_idx]
    opacities = opacities[sort_idx]

    # Prepare canvas
    canvas = bg.expand(H, W, 3).clone()
    transmittance = torch.ones(H, W, 1, device=device)

    # For each splat, draw a 2D Gaussian alpha mask and composite
    for i in range(uv.shape[0]):
        u0, v0 = uv[i]
        su, sv = std_u[i].item(), std_v[i].item()
        if not (math.isfinite(u0.item()) and math.isfinite(v0.item()) and su > 0 and sv > 0):
            continue

        rad_u = max(1, int(coverage * su))
        rad_v = max(1, int(coverage * sv))
        u_min = max(0, int(math.floor(u0.item() - rad_u)))
        u_max = min(W - 1, int(math.ceil(u0.item() + rad_u)))
        v_min = max(0, int(math.floor(v0.item() - rad_v)))
        v_max = min(H - 1, int(math.ceil(v0.item() + rad_v)))
        if u_min >= W or u_max < 0 or v_min >= H or v_max < 0:
            continue

        # Pixel grid in the bbox
        us = torch.arange(u_min, u_max + 1, device=device).view(1, -1)
        vs = torch.arange(v_min, v_max + 1, device=device).view(-1, 1)
        du = (us - u0)
        dv = (vs - v0)
        # Axis-aligned Gaussian (ellipse if su!=sv). For higher fidelity, use full Mahalanobis with Sigma_img^{-1}.
        alpha = torch.exp(-0.5 * ( (du / (su + 1e-6))**2 + (dv / (sv + 1e-6))**2 ))  # (V,U)
        alpha = alpha * opacities[i].clamp(0, 1)
        alpha = alpha.unsqueeze(-1)  # (V,U,1)

        # Composite front-to-back over bbox
        C = colors[i].view(1, 1, 3)
        sub_canvas = canvas[v_min:v_max+1, u_min:u_max+1, :]
        sub_T = transmittance[v_min:v_max+1, u_min:u_max+1, :]

        contrib = sub_T * alpha
        sub_canvas = sub_canvas * (1 - contrib) + C * contrib
        sub_T = sub_T * (1 - alpha)

        # Write back
        canvas[v_min:v_max+1, u_min:u_max+1, :] = sub_canvas
        transmittance[v_min:v_max+1, u_min:u_max+1, :] = sub_T

        # Optional early-out if nearly opaque everywhere
        if transmittance.min() < 1e-3:
            pass

    return canvas.clamp(0, 1)

# ──────────────────────────────────────────────────────────────────────────────
# Demo scene
# ──────────────────────────────────────────────────────────────────────────────

def demo():
    device = get_device()
    H, W = 256, 256

    # Camera intrinsics (simple pinhole)
    fx = fy = 220.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)

    # Camera pose: look at origin from (0, 0, +3)
    eye = torch.tensor([0.0, 0.0, 3.0], device=device)
    at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    view = look_at(eye, at, up)

    # Make a small set of Gaussians in world space
    N = 8
    torch.manual_seed(0)
    means_world = torch.randn(N, 3, device=device)
    means_world[:, 2] = means_world[:, 2].abs() * 0.5  # keep in front of camera (positive Z in world, gets turned to camera space)
    means_world[:, :2] *= 0.5

    sigmas_world = (0.05 + 0.10 * torch.rand(N, device=device))
    colors = torch.rand(N, 3, device=device)
    opacities = 0.5 + 0.5 * torch.rand(N, device=device)

    img = render_gaussians(means_world, sigmas_world, colors, opacities, view, K, H, W)

    img_np = img.detach().cpu().numpy()
    plt.figure(figsize=(4,4))
    plt.imshow(img_np)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# Optional: tiny optimization against a target (single view)
# ──────────────────────────────────────────────────────────────────────────────

def tiny_optimize_demo():
    device = get_device()
    H, W = 128, 128
    fx = fy = 150.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    eye = torch.tensor([0.0, 0.0, 2.5], device=device)
    at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    view = look_at(eye, at, up)

    # Ground-truth scene (3 gaussians)
    with torch.no_grad():
        gt_means = torch.tensor([[ -0.3, -0.15, 0.3],
                                 [  0.2,  0.10, 0.4],
                                 [  0.0, -0.10, 0.6]], device=device)
        gt_sigmas = torch.tensor([0.06, 0.08, 0.05], device=device)
        gt_colors = torch.tensor([[1.0, 0.3, 0.3], [0.3, 1.0, 0.3], [0.3, 0.3, 1.0]], device=device)
        gt_opac = torch.tensor([0.9, 0.8, 0.7], device=device)
        target = render_gaussians(gt_means, gt_sigmas, gt_colors, gt_opac, view, K, H, W)

    # Learnable scene (start from noise)
    means = torch.nn.Parameter(0.3 * torch.randn_like(gt_means))
    sigmas = torch.nn.Parameter(0.05 + 0.02 * torch.rand_like(gt_sigmas))
    colors = torch.nn.Parameter(torch.rand_like(gt_colors))
    opac = torch.nn.Parameter(0.5 * torch.ones_like(gt_opac))

    opt = torch.optim.Adam([means, sigmas, colors, opac], lr=5e-2)

    for it in range(201):
        opt.zero_grad()
        img = render_gaussians(means, sigmas.abs() + 1e-3, colors.sigmoid(), opac.sigmoid(), view, K, H, W)
        loss = F.mse_loss(img, target)
        loss.backward()
        opt.step()
        if it % 50 == 0:
            print(f"it={it:03d}  loss={loss.item():.6f}")

    # Show result vs target
    fig, ax = plt.subplots(1, 2, figsize=(6,3))
    ax[0].imshow(target.detach().cpu().numpy()); ax[0].set_title('Target'); ax[0].axis('off')
    ax[1].imshow(img.detach().cpu().numpy());    ax[1].set_title('Reconstruction'); ax[1].axis('off')
    plt.tight_layout(); plt.show()


def _quick_tests():
    """Very small sanity tests for the renderer.
    These are not exhaustive but catch sign errors and degenerate cases.
    """
    device = get_device()
    H, W = 32, 32
    fx = fy = 50.0
    cx, cy = W / 2.0, H / 2.0
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], device=device, dtype=torch.float32)
    eye = torch.tensor([0.0, 0.0, 3.0], device=device)
    at = torch.tensor([0.0, 0.0, 0.0], device=device)
    up = torch.tensor([0.0, 1.0, 0.0], device=device)
    view = look_at(eye, at, up)

    # Test 1: single centered blob projects near image center and increases intensity over white bg when colored dark
    means = torch.tensor([[0.0, 0.0, 0.5]], device=device)
    sigmas = torch.tensor([0.1], device=device)
    colors = torch.tensor([[0.0, 0.0, 0.0]], device=device)  # black
    opac = torch.tensor([0.9], device=device)
    img = render_gaussians(means, sigmas, colors, opac, view, K, H, W)
    center_px = img[H//2, W//2].mean()
    # On white background, a black blob should reduce brightness at center
    assert center_px < 0.9, f"Center should be darker than background, got {center_px}"

    # Test 2: far blob vs near blob compositing order (near should dominate when overlapping)
    means2 = torch.tensor([[0.0, 0.0, 0.2], [0.0, 0.0, 1.0]], device=device)  # near first
    sigmas2 = torch.tensor([0.1, 0.1], device=device)
    colors2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], device=device)  # red near, blue far
    opac2 = torch.tensor([0.9, 0.9], device=device)
    img2 = render_gaussians(means2, sigmas2, colors2, opac2, view, K, H, W)
    center_rgb = img2[H//2, W//2]
    # Red (near) should dominate over blue (far) at the center
    assert center_rgb[0] > center_rgb[2], f"Near (red) should dominate far (blue): {center_rgb}"

    # Test 3: culling behind camera (point that ends up with negative camera-space Z)
    # With eye=(0,0,3) looking toward the origin and +Z forward in camera space,
    # a world point at z > eye_z (e.g., z=+6.0) lies BEHIND the camera and should be culled.
    means3 = torch.tensor([[0.0, 0.0, 6.0]], device=device)  # behind camera in world coords
    sigmas3 = torch.tensor([0.1], device=device)
    colors3 = torch.tensor([[0.0, 0.0, 0.0]], device=device)
    opac3 = torch.tensor([1.0], device=device)
    img3 = render_gaussians(means3, sigmas3, colors3, opac3, view, K, H, W)
    # Should be identical to background (white)
    assert torch.allclose(img3, torch.ones_like(img3)), "Points behind camera should be culled"

    print("_quick_tests passed.")


if __name__ == "__main__":
    # Run basic sanity tests (do not remove or change unless a test is clearly wrong)
    _quick_tests()
    # Choose which demo to run:
    demo()
    # tiny_optimize_demo()
