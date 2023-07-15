# Parameter list

- [Files](#files)
  - `files:input_directory`: string
  - `files:input_stack`: string
  - `files:input_tlt`: string
  - `files:input_exposure`: string
  - `files:output_directory`: string


- [Tilt Scheme](#tilt-scheme)
  - `tilt_scheme:starting_angle`: float
  - `tilt_scheme:starting_direction`: string, float
  - `tilt_scheme:angle_increment`: float
  - `tilt_scheme:group`: integer
  - `tilt_scheme:exclude_start`: boolean
  - `tilt_scheme:per_view_exposure`: float
  - `angle_offsets:rotation`: float
  - `angle_offsets:tilt`: float
  - `angle_offsets:elevation`: float


- [CTF](#ctf)
  - `ctf:voltage`: float
  - `ctf:amplitude`: float
  - `ctf:cs`: float
  - `ctf:phase_shift`: float


- [Preprocessing](#preprocessing)
  - `preprocessing:run`: boolean
  - `preprocessing:exclude_blank_views`: boolean
  - `preprocessing:exclude_view_indexes`: integer, sequence of integers


- [Alignment](#alignment)
  - `alignment:run`: bool
  - `alignment:rotation_offset`: bool
  - `alignment:tilt_offset`: bool
  - `alignment:elevation_offset`: bool
  - `alignment:pairwise_matching`: bool
  - `alignment:ctf_estimate`: bool
  - `alignment:projection_matching`: bool

- [Compute](#compute)
  - `device`: string
  - `cpu_threads`: integer
  - `verbosity`: string


# Files

### Automatic search using `files:input_directory`

If `files:input_directory` is entered, the program tries to look for the other input files with the basename set as the name of the directory. Otherwise, input files should be entered.

### Input files

- `files:input_stack`: Input stack. Currently, only MRC files are supported.


- `files:input_tlt`: File with the tilt angles of each slice, in degrees, as saved in the input stack. Each slice can be separated with a comma or a new line. This file is optional if the [tilt scheme](#tilt-scheme) is entered. Note that this file takes precedence over the [tilt scheme](#tilt-scheme) entries.


- `files:input_exposure`: File with the accumulated exposure of each slice, in e/A^2, as saved in the input stack. This is used for exposure weighting and is optional. Note that this file takes precedence over the [tilt scheme](#tilt-scheme) entry `per_view_exposure`.

### Output directory

- `files:output_directory`: Directory where every single output is written. Defaults to the current working directory.
  TODO: Add directory structure


# Tilt-scheme

The tilt-scheme includes the number slices, their angles, their order of collection, and optionally their accumulated exposure. There are 2 ways to specify the tilt-scheme. These are presented by priority order.

### Input files

As mentioned in [Files](#files), `files:input_tlt` and `files:input_exposure` can be used to specify the tilt-scheme. These files, if specified, overwrite the `tilt_scheme` parameters.

### `tilt_scheme` parameters

These parameters allow to easily specify the tilt-scheme and is often the simplest option if the exposure and tilt increment are constant through the stack.

- `starting_angle`: The angle, in degrees, of the first collected image. Usually, `0`.
- `starting_direction`: The direction after collecting the first image. This should be `pos`, `positive`, `neg`, `negative`, or a positive or negative number (only the sign is used).
- `angle_increment`: Angle increment, in degrees, between images. Usually, `3`.
- `group`: Number of images that are collected before switching to the opposite side. This refers to the "group"
  for dose-symmetric schemes. For unidirectional schemes, this should correspond to the number of views (there's
  only one group). For bidirectional schemes, this should correspond to the number of views in the first direction.
- `exclude_start`: Whether to exclude the first image from the first group.
- `per_view_exposure`: Per view exposure, in e-/A^2.

Note that these parameters should not account for any known tilt offset added during data collection. Instead, these angle offsets should be specified using the `angles_offset` parameters.

### `angle_offets` parameters

Angle offsets to add to every slice in the stack. See the [geometry conventions](Geometry.md) for more details.

- `rotation`: known offset in around the z-axis, in degrees. Defaults to `0`. This is often refered to as the tilt-axis or rotation angle.
- `tilt`: known offset in around y-axis, in degrees. Defaults to `0`. If any known tilt offset is added during data collection, this is where it should be specified.
- `elevation`:  known offset in around x-axis, in degrees. Defaults to `0`.


# CTF

Microscope parameters. These are used for simulating the CTF during CTF estimate.

- `voltage`: acceleration voltage, in kV. Defaults to `300`.
- `amplitude`: amplitude fraction. Defaults to `0.07`.
- `cs`: Spherical aberration, in micrometers. Defaults to `2.7`.
- `phase_shift`: Phase shift, in degrees. Defaults to `0`.


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

### Angle offsets
- `alignment:rotation_offset`: Whether to search for and/or refine the rotation offset. Defaults to `true`.
- `alignment:tilt_offset`: Whether to search for and/or refine the tilt offset. Defaults to `true`.
- `alignment:elevation_offset`: Whether to search for and/or refine the elevation offset. Defaults to `true`.

### Algorithms

- `alignment:pairwise_matching`: Activate the pairwise matching. Defaults to `true`.
- `alignment:ctf_estimate`: Activate the ctf estimate. Defaults to `true`.
- `alignment:projection_matching`: Activate the projection matching. Defaults to `true`.


# Compute

- `device`: Compute device. Either `auto` (default), `cpu`, `gpu` or `gpu:X`, where `X` is the gpu index, starting from `0`. `auto` will check if there are GPUs on the system, and if so, will use the most free GPU. `gpu` will also use the most free GPU.
- `cpu_threads`: maximum number of CPU threads to use. Defaults to the maximum number of threads available on the system, clamped to `16`.
- `log_level`: Level of logging in the console. `off`, `error`, `info` (default), `trace` or `debug`. Note that log file is at least set to `trace`. `debug` has special meaning and should not be used in production, as it is only intended for debugging. In `debug` mode, a bunch of logs are emitted, possibly including a lot of (sometimes large) files, considerably slowing down the execution and storage footprint.
