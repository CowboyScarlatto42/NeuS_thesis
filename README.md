# NeuS
We present a novel neural surface reconstruction method, called NeuS (pronunciation: /nuːz/, same as "news"), for reconstructing objects and scenes with high fidelity from 2D image inputs.

![](./static/intro_1_compressed.gif)
![](./static/intro_2_compressed.gif)

## [Project page](https://lingjie0206.github.io/papers/NeuS/) |  [Paper](https://arxiv.org/abs/2106.10689) | [Data](https://www.dropbox.com/sh/w0y8bbdmxzik3uk/AAAaZffBiJevxQzRskoOYcyja?dl=0)
This is the official repo for the implementation of **NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction**.

## Usage

#### Data Convention
The data is organized as follows:

```
<case_name>
|-- cameras_xxx.npz    # camera parameters
|-- image
    |-- 000.png        # target image for each view
    |-- 001.png
    ...
|-- mask
    |-- 000.png        # target mask each view (For unmasked setting, set all pixels as 255)
    |-- 001.png
    ...
```

Here the `cameras_xxx.npz` follows the data format in [IDR](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md), where `world_mat_xx` denotes the world to image projection matrix, and `scale_mat_xx` denotes the normalization matrix.

### Setup

Clone this repository

```shell
git clone https://github.com/Totoro97/NeuS.git
cd NeuS
pip install -r requirements.txt
```

<details>
  <summary> Dependencies (click to expand) </summary>

  - torch==1.8.0
  - opencv_python==4.5.2.52
  - trimesh==3.9.8 
  - numpy==1.19.2
  - pyhocon==0.3.57
  - icecream==2.1.0
  - tqdm==4.50.2
  - scipy==1.7.0
  - PyMCubes==0.1.2

</details>

### Running

- **Training without masks**

```shell
python exp_runner.py --mode train --conf ./confs/womask.conf --case <case_name>
```

- **Training with masks**

```shell
python exp_runner.py --mode train --conf ./confs/wmask.conf --case <case_name>
```

- **Extract surface from trained model** 

```shell
python exp_runner.py --mode validate_mesh --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding mesh can be found in `exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply`.

- **View interpolation**

```shell
python exp_runner.py --mode interpolate_<img_idx_0>_<img_idx_1> --conf <config_file> --case <case_name> --is_continue # use latest checkpoint
```

The corresponding image set of view interpolation can be found in `exp/<case_name>/<exp_name>/render/`.

### Uncertainty / geometric sensitivity proxy

This branch contains an experimental pipeline for estimating a geometric uncertainty map from an already trained NeuS model. The goal is not to produce calibrated probabilistic uncertainty, but a local sensitivity proxy: a small dense deformation grid is introduced in the normalized NeuS space `[-1, 1]^3`, and the pipeline measures how much the rendered image would change under infinitesimal perturbations of the grid vertices. Regions where rendering is highly sensitive to local geometric deformation are treated as better constrained by the selected rays; regions with low sensitivity are visualized as less constrained or more uncertain.

The core component is `models/deformation_grid.py`, which defines `DenseDeformationGrid`. The grid stores learnable 3D offsets with shape `[1, 3, R, R, R]` and interpolates offsets at renderer sample points with trilinear interpolation. Points outside the normalized box receive exactly zero deformation. In the renderer (`models/renderer.py`), the grid can be passed through `renderer.render(..., deformation_grid=...)`; SDF query points become `pts + deformation_grid(pts)`, while the rest of the NeuS pipeline remains unchanged. The renderer also supports diagnostic modes:

- `full`: propagates deformation through both the geometry and appearance branches.
- `geometry_only`: isolates the geometry contribution by letting deformation affect weights/opacity/SDF while using detached baseline colors.
- `appearance_only`: isolates the appearance contribution by using detached baseline volumetric weights.
- `appearance_no_normal_grad`: same as `appearance_only`, but without propagating gradients through the normals used by the color network.

The proxy is built as an empirical Hessian-like diagonal/trace. For each selected ray and each RGB channel, the scalar rendered output is differentiated with respect to the grid offsets, then squared element-wise and accumulated:

```text
H_proxy(voxel, component) += (d RGB_channel / d offset_component)^2
H_trace(voxel) = sum_components H_proxy(voxel, component)
```

This avoids summing RGB channels before differentiation, because `grad(rgb.sum())^2` would introduce unwanted cross terms. Before the computation, the NeuS model is frozen (`eval()` and `requires_grad_(False)` for the SDF, deviation, color network, and outside NeRF modules): gradients are used only to measure sensitivity with respect to the deformation grid.

#### Deformation grid smoke test

To verify that the deformation grid is correctly connected to the renderer:

```shell
python custom_codes/uncertainty/deformation_grid_smoke_test.py \
  --conf ./confs/long_test.conf \
  --case <case_name> \
  --candidate_rays 128 \
  --num_rays 16 \
  --grid_resolution 16
```

The script chooses foreground rays from the mask, renders a baseline without the grid, renders again with a zero-initialized grid, and checks two conditions: the RGB difference between the baseline and the zero grid must be negligible, and the gradient with respect to the grid offsets must be non-zero. It also prints opacity statistics for the selected rays and the gradient shape/magnitude.

#### Single-image accumulation

To produce a first proxy map from a single view:

```shell
python custom_codes/uncertainty/accumulate_hessian_proxy.py \
  --conf ./confs/long_test.conf \
  --case <case_name> \
  --img_idx 0 \
  --candidate_rays 512 \
  --num_rays 128 \
  --grid_resolution 32 \
  --output_dir exp/<case_name>/uncertainty_single_view \
  --seed 42 \
  --prefix_counts 8,16,32,64
```

The script selects candidates from the mask, keeps only rays with `weight_sum > --opacity_threshold`, computes RGB gradients with respect to the grid, and saves:

- `hessian_proxy_components.npy`: per-component contributions `[3, R, R, R]`.
- `hessian_proxy_trace.npy`: scalar trace `[R, R, R]`.
- `hessian_proxy_metadata.json`: configuration, checkpoint, grid statistics, and opacity statistics.
- `selected_rays_metadata.json`: selected pixels, linear indices, and opacity values.
- `per_ray_hessian_contributions.csv/json`: total contribution, maximum single-voxel contribution, and R/G/B contributions for each ray.
- optional `hessian_proxy_*_prefix_<N>.npy` files when `--prefix_counts` is used, useful for checking convergence as more rays are accumulated.

#### Sensitivity mode comparison

To understand whether the proxy is dominated by the geometry branch or the appearance branch:

```shell
python custom_codes/uncertainty/compare_sensitivity_modes.py \
  --conf ./confs/long_test.conf \
  --case <case_name> \
  --img_idx 0 \
  --candidate_rays 512 \
  --num_rays 128 \
  --grid_resolution 32 \
  --modes full,geometry_only,appearance_only,appearance_no_normal_grad \
  --output_dir exp/<case_name>/uncertainty_modes \
  --seed 42
```

Each mode writes a subdirectory with `hessian_proxy_components.npy`, `hessian_proxy_trace.npy`, `metadata.json`, and per-ray contributions. The top-level output directory contains `sensitivity_modes_summary.json/csv`. The script also includes a regression check: `diagnostic_mode='full'` must match the standard zero-grid RGB pipeline within `--rgb_tolerance`.

#### Geometry-oriented multi-view accumulation

The most useful version for the thesis is the multi-view accumulation in `geometry_only` mode, because it separates geometric sensitivity from color sensitivity as much as possible:

```shell
python custom_codes/uncertainty/accumulate_multiview_geometry_proxy.py \
  --conf ./confs/long_test.conf \
  --case <case_name> \
  --image_indices 0,10,20,30,40,50,61,75,90,105,120 \
  --rays_per_image 64 \
  --grid_resolution 32 \
  --output_dir exp/<case_name>/uncertainty_multiview_geometry \
  --seed 42
```

For each image, the script samples `--rays_per_image` foreground pixels without replacement, accumulates squared RGB gradients, and adds the contributions to the same global grid. Indices `0-60` are annotated as `orbit_1`, and indices `61-127` as `orbit_2`, making it possible to inspect how much each orbit contributes. The main outputs are:

- `hessian_geometry_components.npy`: vector grid `[3, R, R, R]`.
- `hessian_geometry_raw.npy`: scalar grid `[R, R, R]`, obtained by summing the three components.
- `multiview_geometry_proxy_metadata.json`: checkpoint, selected images, orbit assignment, global statistics, seed, device, and elapsed time.
- `selected_rays_metadata.json`: selected pixels for each image.
- `per_image_stats.csv`: per-image contribution and cumulative contribution.

#### Mesh visualization

Once a NeuS mesh has been extracted in normalized coordinates, the proxy grid can be interpolated at mesh vertices and exported as colored PLY files:

```shell
python custom_codes/uncertainty/export_geometry_proxy_colored_mesh.py \
  --hessian_grid exp/<case_name>/uncertainty_multiview_geometry/hessian_geometry_raw.npy \
  --reconstruction_mesh exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply \
  --output_dir exp/<case_name>/uncertainty_colored_mesh
```

The script requires at least 99% of vertices to lie inside `[-1, 1]^3`; if this check fails, the reconstruction mesh is likely not in normalized NeuS coordinates. It produces:

- `mesh_geometry_proxy_confidence.ply`: `viridis` colors over `log10(H + eps)`.
- `mesh_geometry_proxy_inverse_sensitivity.ply`: `magma` colors over `-log10(H + eps)`, useful as a qualitative visualization of less constrained regions.
- `vertex_proxy_values.csv`: raw, logarithmic, and inverse-sensitivity values for every vertex.
- `confidence_colorbar.png` and `inverse_sensitivity_colorbar.png`.
- `colored_mesh_metadata.json`: percentile limits, statistics, and applied transformations.

Percentile clipping (`--lower_percentile`, `--upper_percentile`) affects colors only, not the values saved in the CSV file.

#### Validation against the ground-truth mesh

To evaluate whether the proxy correlates with the real geometric error:

```shell
python custom_codes/uncertainty/validate_geometry_proxy_on_mesh.py \
  --hessian_grid exp/<case_name>/uncertainty_multiview_geometry/hessian_geometry_raw.npy \
  --reconstruction_mesh exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply \
  --gt_mesh <path_to_gt_mesh.ply> \
  --conf ./confs/long_test.conf \
  --case <case_name> \
  --num_surface_points 50000 \
  --output_dir exp/<case_name>/uncertainty_validation \
  --seed 42
```

The script samples points on the reconstructed surface, interpolates the proxy in normalized NeuS coordinates, maps the points to the ground-truth frame using the NeuS `scale_mat_0` convention, measures point-to-surface distance to the GT mesh, and computes:

- Spearman correlations between geometric error and `H` or `-H`.
- sparsification curves comparing proxy-guided removal, a random baseline, and an oracle.
- AUC values up to `--auc_max_removed_fraction`.
- error statistics by proxy quartile.

The outputs are `spearman_summary.json`, `sampled_mesh_points_with_proxy.csv`, `sparsification_curve.csv/png`, and `quartile_error_summary.csv`.

#### Stability with respect to the sampling seed

To compare two proxy maps obtained with different sampling seeds:

```shell
python custom_codes/uncertainty/compare_proxy_seed_maps.py \
  --hessian_grid_a exp/<case_name>/uncertainty_multiview_geometry_seed42/hessian_geometry_raw.npy \
  --hessian_grid_b exp/<case_name>/uncertainty_multiview_geometry_seed123/hessian_geometry_raw.npy \
  --label_a seed_42 \
  --label_b seed_123 \
  --reconstruction_mesh exp/<case_name>/<exp_name>/meshes/<iter_steps>.ply \
  --num_surface_points 50000 \
  --lowest_confidence_fraction 0.25 \
  --output_dir exp/<case_name>/uncertainty_seed_comparison
```

The script saves `seed_map_comparison_summary.json`, `seed_map_comparison_points.csv`, and a scatter plot `seed_map_comparison_scatter.png`. The main metrics are Spearman correlation between the two grids, Pearson correlation on `log10(H + eps)`, IoU, and overlap for the lowest-confidence point subset.

#### Practical notes

- The pipeline requires an already trained NeuS checkpoint and internally loads it with `--is_continue=True`.
- Grid coordinates are normalized NeuS coordinates, not direct CORTO/GT coordinates.
- `hessian_geometry_raw.npy` and `hessian_proxy_trace.npy` are sensitivity proxies, not calibrated Bayesian variances.
- High `H` values indicate high rendering sensitivity to local geometric perturbations; for an uncertainty-style visualization, `-log10(H + eps)` is often used.
- Increasing `--grid_resolution`, `--num_rays`, `--rays_per_image`, or the number of images improves coverage but quickly increases memory and runtime, because each RGB channel requires a separate `torch.autograd.grad` call.

### Train NeuS with your custom data

More information can be found in [preprocess_custom_data](https://github.com/Totoro97/NeuS/tree/main/preprocess_custom_data).

## Citation

Cite as below if you find this repository is helpful to your project:

```
@article{wang2021neus,
  title={NeuS: Learning Neural Implicit Surfaces by Volume Rendering for Multi-view Reconstruction},
  author={Wang, Peng and Liu, Lingjie and Liu, Yuan and Theobalt, Christian and Komura, Taku and Wang, Wenping},
  journal={arXiv preprint arXiv:2106.10689},
  year={2021}
}
```

## Acknowledgement

Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr) and [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch). Thanks for these great projects.
