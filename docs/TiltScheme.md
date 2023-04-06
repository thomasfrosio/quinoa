## `Tilt Scheme`

The tilt-scheme includes the number tilt images, their order of collection and their exposure. There are 3 ways to
specify the tilt-scheme. These are presented by priority order.

### `.mdoc file`

If a `.mdoc` file is provided, it is parsed and the tilt-scheme is extracted. This file must of course match the 
input stack, which can be tricky if [views are to be excluded by their indexes](ExcludeViews.md).

### `.tlt and .exposure files`

A simpler `.tlt` and `.exposure` file pair can be used instead. The `.tlt` file should be a new-line separated list 
of the tilt angles as saved in the input stack file. The `.exposure` file contains the e-/A^2 exposure instead.

### `order_* parameters`

If the `.mdoc` file is found, or if the `.tlt` and `.exposure` files are found, the `order_*` entries are not used.
Otherwise, every `order_*` entry below should be specified.

These parameters allow to construct the tilt-scheme without requiring any input file and is the preferred option if 
the exposure and tilt angle is constant throughout the tilt-series.

- `order_starting_angle`: The angle, in degrees, of the first collected image. Usually, `0`.
- `order_starting_direction`: The direction after collecting the first image. This should be `pos`, `positive`, 
  `neg` or `negative`.
- `order_angle_increment`: Angle increment, in degrees, between images. Usually, `3`.
- `order_group`: Number of images that are collected before switching to the opposite side. This refers to the "group"
  for dose-symmetric schemes. For unidirectional schemes, this should correspond to the number of views (there's 
  only one group). For bidirectional schemes, this should correspond to the number of views in the first direction.
- `order_exclude_start`: Exclude the first image from the first group.
- `order_per_view_exposure`: Per view exposure, in e-/A^2.
