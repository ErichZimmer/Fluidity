# A GUI for OpenPIV-python

A Graphical User Interface (GUI), providing fast and efficient workflow for evaluating and post-processing particle image velocimetry (PIV) images or results. This repository is based off of [openpiv_tk_gui](https://github.com/OpenPIV/openpiv_tk_gui).
![Screen shot of the GUI showing a vector plot.](https://raw.githubusercontent.com/ErichZimmer/openpiv-python-gui/master/fig/piv_challenge_2014_case_b.JPG)

## Warning
This GUI was developed hastily for a PIV project and thus has many bugs. Additionally, the GUI may not be tailored to your needs as it has features that was found useful for an in-house PIV system. 

## Features
+ Works with a wide range of image types (RGB converted to grayscale using 0.299R 0.587B 0.144G)
+ Create movies (mp4, avi, non-optimized gif)
+ Process movies (converts to images)
+ Custom image frequencing
+ Extensive image pre-processing with batch image processing (saves as 8 bit images)
+ Interactive calibration
+ High precision processing using iterative image deformation
+ 3 subpixel estimators (gaussian, parabolic, centroid)
+ Ensemble correlation with iterative image deformation
+ Rectangular interrogation windows supported (25% slower)
+ Average all results (excluding ensemble correlation)
+ Parallel batch processing
+ Extensive post-processing
+ Import vectors for post-processing
+ Synthetic image generation
+ No proprietary environments

[Example output](https://user-images.githubusercontent.com/69478071/140243359-f234c093-4ce6-49d5-ae61-f1bc684de042.mp4)

## Documentation 

See wiki

## To-do list
- [ ] second peak search
- [ ] nxn 2D subpixel estimator
- [ ] direct correlation for limited search area
- [ ] precomputed dense predictor for first pass
- [ ] temporal background removal
- [ ] Uncertainty quantification
- [ ] proper orthogonal decomposition
- [ ] image rectification (calibration)
- [ ] pressure estimation
- [ ] Add stereo or other 3D processing options (maybe a new GUI for this too)
- [ ] do all PIV processing in c (including parallel)

## References
+ [openpiv_tk_gui](https://github.com/OpenPIV/openpiv_tk_gui) as a starting platform
+ [openpiv-python](https://github.com/OpenPIV/openpiv-python) as core processor
+ [PIVlab](https://github.com/Shrediquette/PIVlab) for synthetic image generation and auto scaling of vectors 
+ [Particle Image Velocimetry: A Practical Guide](https://link.springer.com/book/10.1007/978-3-662-03637-2) got me started in PIV 
