## `Exclude views`

The preprocessing stage can exclude views from the stack. This is controlled by the `preprocessing_exclude_*` parameters.

### `Exclude views based on their indexes`

If we know the views indexes beforehand, the best is to either:

- Specify the indexes (starting from 0) of the views to exclude in the `preprocessing_exclude_indexes` parameter.
  For instance, a value of `[0, 39, 40]` would remove the views at index 0, 39, and 40.


- Or, remove the views from _all_ input files before running the alignment. However, this option is _not_ supported with
  the `order_` parameters, so another input mode for the tilt-scheme should be used. See [TiltScheme.md](.
  /TiltScheme.md).


### `Exclude "bad" views automatically`

The preprocessing stage can run a mass-normalization function and try to identify images that are far off the rest 
of the stack. This is based on very simple statistics features (mean, median, etc.) and whilst it may not be reliable
for all use-cases, it should be enough to remove blank images. Use `preprocessing_exclude_blank_views: True` to turn 
this feature on. Note that it can be used in addition of `preprocessing_exclude_indexes`.
