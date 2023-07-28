# Parameter list

- [Files](#files)
  - `files:input_directory`: string
  - `files:input_stack`: string
  - `files:input_tlt`: string
  - `files:input_exposure`: string
  - `files:output_directory`: string


- [Tilt Scheme](#tilt-scheme)
  - `tilt_scheme:order:starting_angle`: real
  - `tilt_scheme:order:starting_direction`: string, real
  - `tilt_scheme:order:angle_increment`: real
  - `tilt_scheme:order:group`: integer
  - `tilt_scheme:order:exclude_start`: boolean
  - `tilt_scheme:order:per_view_exposure`: real
  - `tilt_scheme:rotation_offset`: real
  - `tilt_scheme:tilt_offset`: real
  - `tilt_scheme:elevation_offset`: real
  - `tilt_scheme:voltage`: real
  - `tilt_scheme:amplitude`: real
  - `tilt_scheme:cs`: real
  - `tilt_scheme:phase_shift`: real
  - `tilt_scheme:astigmatism_value`: real
  - `tilt_scheme:astigmatism_angle`: real


- [Preprocessing](#preprocessing)
  - `preprocessing:run`: boolean
  - `preprocessing:exclude_blank_views`: boolean
  - `preprocessing:exclude_view_indexes`: integer, sequence of integers


- [Alignment](#alignment)
  - `alignment:run`: bool
  - `alignment:fit_rotation_offset`: bool
  - `alignment:fit_tilt_offset`: bool
  - `alignment:fit_elevation_offset`: bool
  - `alignment:fit_phase_shift`: bool
  - `alignment:fit_astigmatism`: bool
  - `alignment:use_initial_pairwise_alignment`: bool
  - `alignment:use_ctf_estimate`: bool
  - `alignment:use_projection_matching`: bool

- [Compute](#compute)
  - `compute:device`: string
  - `compute:n_cpu_threads`: integer
  - `compute:verbosity`: string


# Files

### Automatic search using `files:input_directory`

If `files:input_directory` is entered, the program tries to look for the other input files with the basename set as the name of the directory. Otherwise, input files should be entered.

### Input files

- `files:input_stack`: Input stack. Currently, only MRC files are supported.


- `files:input_tlt`: File with the tilt angles of each slice, in degrees, as saved in the input stack. Each slice should be separated by a new line. This file is optional if the [tilt scheme:order](#tilt-scheme) parameters are entered. Note that this file takes precedence over the [tilt scheme:order](#tilt-scheme) entries.


- `files:input_exposure`: File with the accumulated exposure of each slice, in e/A^2, as saved in the input stack. This is used for exposure weighting and is optional. It should have the same format as `files:input_tlt`. Note that this file takes precedence over the [tilt scheme:order](#tilt-scheme) entry `per_view_exposure`.

### Output directory

- `files:output_directory`: Directory where every single output is written. Defaults to the current working directory.
  TODO: Add directory structure


# Tilt-scheme

The tilt-scheme includes the number slices, their angles, their order of collection, optionally their accumulated exposure, and everything related to the data acquisition (e.g. voltage, cs, etc.). There are 2 ways to specify the tilt-scheme.

### Input files

As mentioned in [Files](#files), `files:input_tlt` and `files:input_exposure` can be used to specify the `tilt_scheme:order` parameters. These files, if specified, overwrite the `tilt_scheme:order` parameters.

### `tilt_scheme:order` parameters

These parameters allow to easily specify the collection order and is often the simplest way to start if the exposure and tilt increment are constant through the stack.

- `tilt_scheme:order:starting_angle`: The angle, in degrees, of the first collected image. Usually, `0`.
- `tilt_scheme:order:starting_direction`: The direction after collecting the first image. This should be `pos`, `positive`, `neg`, `negative`, or a positive or negative number (only the sign is used).
- `tilt_scheme:order:angle_increment`: Angle increment, in degrees, between images. Usually, `3`.
- `tilt_scheme:order:group`: Number of images that are collected before switching to the opposite side. This refers to the "group"
  for dose-symmetric schemes. For unidirectional schemes, this should correspond to the number of views (there's
  only one group). For bidirectional schemes, this should correspond to the number of views in the first direction.
- `tilt_scheme:order:exclude_start`: Whether to exclude the first image from the first group.
- `tilt_scheme:order:per_view_exposure`: Per view exposure, in e-/A^2.

Note that these parameters should not account for any known tilt offset added during data collection. Instead, these angle offsets should be specified using the `tilt_scheme:*_offset` parameters.

### `tilt_scheme:*_offets` parameters

Angle offsets to add to every slice in the stack. See the [geometry conventions](Geometry.md) for more details.

- `tilt_scheme:rotation_offset`: known offset in around the z-axis, in degrees. Defaults to `0`. This is often refered to as the tilt-axis or rotation angle.
- `tilt_scheme:tilt_offset`: known offset in around y-axis, in degrees. Defaults to `0`. If any known tilt offset is added during data collection, this is where it should be specified.
- `tilt_scheme:elevation_offset`:  known offset in around x-axis, in degrees. Defaults to `0`.


### CTF-related parameters

Microscope parameters. These are used for simulating the CTF during CTF estimate. See the [ctf conventions](CTF.md) for more details.

- `tilt_scheme:voltage`: acceleration voltage, in kV. Defaults to `300`.
- `tilt_scheme:amplitude`: amplitude fraction. Defaults to `0.07`.
- `tilt_scheme:cs`: Spherical aberration, in micrometers. Defaults to `2.7`.
- `tilt_scheme:phase_shift`: Phase shift, in degrees. Defaults to `0`.
- `tilt_scheme:astigmatism_value`: Known astigmatism value, in micrometers. This is usually not necessary, but if for some reason the data has a known and significant astigmatism, specifying it here may help the fitting. Default to `0`.
- `tilt_scheme:astigmatism_angle`: Known astigmatism angles, in degrees. This is usually not necessary, but if for some reason the data has a known and significant astigmatism, specifying it here may help the fitting. Default to `0`.

# Preprocessing

The preprocessing steps can be turned off using `preprocessing:run: false` (defaults to `true`).

## Exclude views

The preprocessing stage can exclude views from the input stack. This is controlled by the `preprocessing:exclude_blank_views` and `preprocessing:exclude_view_indexes` parameters.

### Exclude views based on their indexes

If we know the view indexes beforehand, the best is to either:

- Specify the indexes (starting from 0) of the views to exclude in the `preprocessing:exclude_view_indexes` parameter. Should be a single value or a sequence `[...]` of values.  For instance, a value of `[0, 39, 40]` would remove the views at index 0, 39, and 40.


- Or, remove the views from _all_ input files before running the program. Note however that this option is _not_ supported with the [tilt scheme](#tilt-scheme) parameters because the program cannot know which views were removed from the stack, so another input mode for the tilt-scheme should be used.

### Exclude "bad" views automatically

The preprocessing stage can run a mass-normalization function and try to identify images that are far off from the rest of the stack. This is based on basic statistics features (mean, median, etc.) and whilst it may not be reliable for all use-cases, it should be enough to remove blank images, which is the main purpose of this step. Use `preprocessing:exclude_blank_views: true` to turn this feature on. Note that this step can be used in addition of `preprocessing:exclude_view_indexes`.


# Alignment

The alignments can be turned off using `alignment:run: false` (defaults to `true`).

- `alignment:fit_rotation_offset`: Whether to search for and/or refine the rotation offset. Defaults to `true`.
- `alignment:fit_tilt_offset`: Whether to search for and/or refine the tilt offset. Defaults to `true`.
- `alignment:fit_elevation_offset`: Whether to search for and/or refine the elevation offset. Defaults to `true`.
- `alignment:fit_phase_shift`: Whether to search for and/or refine the phase shift. Defaults to `false`.
- `alignment:fit_astigmatism`: Whether to search for and/or refine the astigmatism. Defaults to `true`.


- `alignment:use_initial_pairwise_alignment`: Activate the initial pairwise matching alignment. This searches for the per-slice shifts and can fit and/or refine the rotation offset. Defaults to `true`. If `false`, the input tilt-series should already roughly be aligned.
- `alignment:use_ctf_estimate`: Activate the ctf estimate. This searches for the angle offsets and the per-slice defocus, refines the per-slice shifts with the pairwise cosine-stretching alignment (using the new angle offsets) and can also fit and/or refine the phase shift and global astigmatism. Defaults to `true`.
- `alignment:use_projection_matching`: Activate the projection matching. Defaults to `true`.


# Compute

- `compute:device`: Compute-device. Either `auto` (default), `cpu`, `gpu` or `gpu:X`, where `X` is the gpu index, starting from `0`. `auto` will check if there are GPUs on the system, and if so, will use the most free GPU. `gpu` will also use the most free GPU.
- `compute:n_cpu_threads`: maximum number of CPU threads to use. Defaults to the maximum number of threads available on the system, clamped to `16`.
- `compute:log_level`: Level of logging in the console. `off`, `error`, `info` (default), `trace` or `debug`. Note that the log file is at least set to `trace`. `debug` has special meaning and should not be used in production, as it is only intended for debugging. In `debug` mode, a bunch of logs are emitted, possibly including a lot of (sometimes large) files, considerably slowing down the execution and storage footprint.
