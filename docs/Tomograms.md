## `Tomogram reconstruction algorithm`

The tomograms we generate are meant for visualization (+denoising, segmentation, ...); they are not meant for subtomogram averaging. As such,


### `Fourier cubes vs real-space backprojection`

To reconstruct the entire field-of-view, Fourier cubes are not ideal mostly because within a cube the motion is uniform. To render a warping over the entire FOV, we have to stitch together small enough cubes so that 

Within a cube, the motion is uniform. While cubes can be 
When it comes to filtering (which includes CTF correction), one of the best approach is to
