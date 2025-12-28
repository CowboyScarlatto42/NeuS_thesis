# ============================================================
# Goal:
# 1) Verify if dataset views are naturally ordered.
# 2) Build a realistic "orbit-like" view subset by approximating an NMC safety ellipse
#    and matching the closest available views from the dataset.
# 3) Save nested splits (4 ⊂ 8 ⊂ 16 ⊂ 32) and visualize the result.
#
# Notes:
# - We only assume cameras_spe3r.npz contains world_mat_i and scale_mat_i.
# - Object center is approximated by the centroid of camera centers.
# - Orbit plane is estimated by PCA on camera centers.
# - NMC orbit is embedded in that plane and matched to the nearest dataset views.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt

# ---- paths ----
NPZ_PATH  = "/content/drive/MyDrive/Tesi/neus/serious_analysis/data/hst_neus/cameras_spe3r.npz"
SPLIT_DIR = "/content/drive/MyDrive/Tesi/neus/serious_analysis/splits"

# ---- settings ----
SIZES = [4, 8, 16, 32]
LABEL_STEP_TEXT = 40   # controls how many labels to show in the "ordering" plot

# ============================================================
# Part A — Load cameras and show that the dataset is not ordered
# ============================================================

def camera_center_from_P_svd(P3x4):
    # Camera center C satisfies P @ [C;1] = 0 (homogeneous nullspace)
    _, _, Vt = np.linalg.svd(P3x4)
    C_h = Vt[-1]
    if abs(C_h[-1]) < 1e-12:
        return C_h[:3]
    C_h = C_h / C_h[-1]
    return C_h[:3]

def load_centers_and_dirs(npz_path):
    """
    Returns:
      ids: (N,) integer view indices (e.g., 0..999)
      centers: (N,3) camera centers in dataset world frame
      dirs: (N,3) approximate viewing directions (camera optical axis) in world frame
    """
    data = np.load(npz_path, allow_pickle=True)
    ids = sorted([int(k.split("_")[-1]) for k in data.keys() if k.startswith("world_mat_")])
    if len(ids) == 0:
        raise RuntimeError("No world_mat_* keys found in NPZ.")

    centers, dirs = [], []
    for i in ids:
        W = data[f"world_mat_{i}"]
        S = data[f"scale_mat_{i}"]
        P4 = W @ S
        P  = P4[:3, :4].astype(np.float64)

        # camera center
        C = camera_center_from_P_svd(P)
        centers.append(C)

        # viewing direction (approx): optical axis in world coords
        # This is consistent with what we've been using for angular plots.
        R = P4[:3, :3].astype(np.float64)
        v = -R.T @ np.array([0., 0., 1.], dtype=np.float64)
        v = v / (np.linalg.norm(v) + 1e-12)
        dirs.append(v)

    return np.array(ids), np.stack(centers, axis=0), np.stack(dirs, axis=0)

# --- load ---
ids, centers, vdirs = load_centers_and_dirs(NPZ_PATH)
N = len(ids)
print(f"[INFO] Loaded cameras: {N} views. id range: {ids.min()}..{ids.max()}")
print("[OK] centers shape:", centers.shape, "| vdirs shape:", vdirs.shape)

# --- plot: centers colored by view index (to check ordering) ---
idx = np.arange(N)

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")
sc = ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], c=idx, s=8, alpha=0.9)
ax.plot(centers[:, 0], centers[:, 1], centers[:, 2], linewidth=0.8, alpha=0.6)

step = max(1, N // LABEL_STEP_TEXT)
for j in range(0, N, step):
    ax.text(centers[j, 0], centers[j, 1], centers[j, 2], str(j), fontsize=7)

ax.set_title("Camera centers colored by dataset index (ordering check)")
ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
cbar = plt.colorbar(sc, ax=ax, fraction=0.03, pad=0.1)
cbar.set_label("View index")
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()

# --- numeric check: distance between consecutive indices ---
d = np.linalg.norm(centers[1:] - centers[:-1], axis=1)
print(f"[INFO] Consecutive-center distance: mean={float(d.mean()):.3f} | std={float(d.std()):.3f}")

plt.figure(figsize=(8, 3))
plt.plot(d)
plt.title("Distance between consecutive camera centers (index i -> i+1)")
plt.xlabel("i (between i and i+1)")
plt.ylabel("distance")
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n[CONCLUSION] If the curve above looks chaotic and the distances fluctuate strongly,")
print("            the dataset indices are NOT a meaningful 'orbit order'.\n")

# ============================================================
# Part B — Orbit-like selection using NMC safety ellipse (On Optimal Observation Orbits for Learning Gaussian
#                                                        Splatting-based3D Modelsof Unknown Resident Space Objects)
# ============================================================

# If a satellite moving around the object has taken the images, what would be its orbit plane?
# Principal Component Analysis (PCA) to find best axis of camera centers
# Axis 1: direction of maximum variance
# Axis 2: direction of second maximum variance (orthogonal to axis 1)
# Axis 3: direction of minimum variance (orthogonal to axes 1 and 2)
def pca_plane_basis(X):
    """
    Estimate a plausible 'orbit plane' from camera centers.
    Returns two orthonormal basis vectors spanning the best-fit plane.
    """
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    e1 = Vt[0] / np.linalg.norm(Vt[0])
    e2 = Vt[1] / np.linalg.norm(Vt[1])
    return e1, e2

def nmc_points(k, delta_x, omega=1.0):
    """
    Natural Motion Circumnavigation (NMC) from the paper:
      x(t) = -(Δx/2) cos(ωt)
      y(t) = 0
      z(t) =  (Δx/4) sin(ωt)
    """
    t = np.linspace(0, 2*np.pi, k, endpoint=False)
    x = -(delta_x / 2.0) * np.cos(omega * t)
    z =  (delta_x / 4.0) * np.sin(omega * t)
    return x, z

def look_at_dirs(positions, target):
    # desired camera direction: point toward target center
    v = target[None, :] - positions
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    return v

def greedy_match(centers, dirs, desired_pos, desired_dir, wp=1.0, wa=1.0):
    """
    For each desired orbit point, pick the closest dataset view (no repeats) using:
      score = wp * ||C - C*|| + wa * angle(v, v*)
      C, v: camera center, viewing direction of the dataset view
      C*, v*: desired center and direction on orbit
    """
    N = centers.shape[0]
    unused = np.ones(N, dtype=bool)
    chosen = []

    for t in range(desired_pos.shape[0]):
        dp = np.linalg.norm(centers - desired_pos[t], axis=1)
        cosang = np.clip(dirs @ desired_dir[t], -1.0, 1.0)
        da = np.arccos(cosang)
        score = wp * dp + wa * da
        score[~unused] = np.inf
        j = int(np.argmin(score))
        chosen.append(j)
        unused[j] = False

    return chosen

def viewname(i):
    return f"{int(i):03d}.png"

def nested_from_32(sequence32, k):
    # deterministic nested subsets from the 32-point sequence
    step = len(sequence32) // k
    return [sequence32[i] for i in range(0, len(sequence32), step)][:k]

def generate_nmc_splits(ids, centers, vdirs, output_prefix):
    if ids.size == 0:
        raise ValueError(f"Empty subset for prefix {output_prefix}")

    print(f"\n[INFO] Generating NMC splits for {output_prefix}: {ids.size} views (id range {ids.min()}..{ids.max()})")

    # --- define target and plane ---
    target = centers.mean(axis=0)          # proxy for object center
    X = centers - target
    e1, e2 = pca_plane_basis(X)

    # --- choose Δx based on dataset scale (safety ellipse on the point cloud sphere) ---
    r = np.linalg.norm(centers - target, axis=1)
    r_med = float(np.median(r))
    delta_x = 2.0 * r_med  # because NMC amplitude in x is Δx/2 ~ r_med
    print(f"[INFO] NMC: auto Δx = {delta_x:.3f} (median radius {r_med:.3f})")

    # --- matching weights (position vs angle) ---
    wp = 1.0
    wa = r_med   # 1 rad (~57°) ~ cost of moving ~r_med in space
    print(f"[INFO] Matching weights: wp={wp} | wa={wa:.3f}")

    # --- build desired orbit (k=32), embed in dataset plane ---
    KMAX = max(SIZES)
    x, z = nmc_points(KMAX, delta_x, omega=1.0)
    desired_pos_32 = target[None, :] + x[:, None] * e1[None, :] + z[:, None] * e2[None, :]
    desired_dir_32 = look_at_dirs(desired_pos_32, target)

    # --- match desired points to nearest dataset views ---
    chosen_idx_32 = greedy_match(centers, vdirs, desired_pos_32, desired_dir_32, wp=wp, wa=wa)
    chosen_view_ids_32 = ids[chosen_idx_32]

    # --- save nested splits ---
    os.makedirs(SPLIT_DIR, exist_ok=True)
    for k in SIZES:
        subset = nested_from_32(chosen_view_ids_32, k)
        out = os.path.join(SPLIT_DIR, f"{output_prefix}_nmc_{k}.txt")
        with open(out, "w") as f:
            for vid in subset:
                f.write(viewname(vid) + "\n")
        print(f"[OK] wrote {out} ({k} views)")

    print(f"[DONE] NMC orbit-like nested splits generated for {output_prefix}.\n")

    # ============================================================
    # Part C — Visual validation of orbit-like selection
    # ============================================================

    selected_idx_32 = [np.where(ids == vid)[0][0] for vid in chosen_view_ids_32]
    selected_centers_32 = centers[selected_idx_32]
    selected_dirs_32 = vdirs[selected_idx_32]

    # --- plot: orbit in 3D + selected centers ---
    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2], s=5, alpha=0.2, label="All camera centers")
    ax.plot(desired_pos_32[:, 0], desired_pos_32[:, 1], desired_pos_32[:, 2], "k--", linewidth=2, label="Desired NMC orbit")
    ax.scatter(selected_centers_32[:, 0], selected_centers_32[:, 1], selected_centers_32[:, 2], c="orange", s=50, label="Selected views (k=32)")
    ax.scatter(target[0], target[1], target[2], c="red", s=80, marker="*", label="Target center")

    ax.set_title(f"NMC orbit-like view selection ({output_prefix})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_box_aspect([1, 1, 1])
    plt.tight_layout()
    plt.show()

    # --- plot: continuity (distance between consecutive selected views) ---
    dist_seq = np.linalg.norm(selected_centers_32[1:] - selected_centers_32[:-1], axis=1)

    plt.figure(figsize=(7, 3))
    plt.plot(dist_seq, "-o", markersize=3)
    plt.xlabel("Step along orbit")
    plt.ylabel("Distance between consecutive views")
    plt.title(f"Continuity check along NMC selection ({output_prefix})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- plot: viewing directions on unit sphere (sanity check) ---
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(vdirs[:, 0], vdirs[:, 1], vdirs[:, 2], s=5, alpha=0.2, label="All viewing directions")
    ax.scatter(selected_dirs_32[:, 0], selected_dirs_32[:, 1], selected_dirs_32[:, 2], c="orange", s=50, label="Selected directions")

    ax.set_title(f"Viewing directions (unit sphere) — {output_prefix}")
    ax.set_box_aspect([1, 1, 1])
    ax.legend()
    plt.tight_layout()
    plt.show()


black_mask = (ids >= 0) & (ids <= 500)
earth_mask = (ids >= 501) & (ids <= 999)

generate_nmc_splits(ids[black_mask], centers[black_mask], vdirs[black_mask], "hst_black")
generate_nmc_splits(ids[earth_mask], centers[earth_mask], vdirs[earth_mask], "hst_earth")

print("[FINAL CHECK]")
print("- First plots: confirm dataset index is not an orbit order.")
print("- NMC plots: confirm selected views follow an orbit-like path with reasonable continuity.")
