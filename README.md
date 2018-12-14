## DeepFocus

This repository provides source code, network models and dataset of the DeepFocus project.

The source code and network models are implemented with TensorFlow. To further improve the method's runtime performance, we recommond use of NVIDIA TensorRT toolbox to optimize the network inferences with 16-bit precision on your machines. We observed no more than 0.01dB (PSNR) quality drop while about 8 times speed up after the inference optimization.   

If you use our code, please cite our paper: Lei Xiao, Anton Kaplanyan, Alexander Fix, Matt Chapman, Douglas Lanman, "DeepFocus: Learned Image Synthesis For Computational Displays", SIGGRAPH Asia 2018 Technical Paper.

The technical paper, video and more supplementary materials can be found at: https://research.fb.com/publications/deepfocus-learned-image-synthesis-for-computational-displays/

## License
DeepFocus is CC-BY-NC 4.0 (FAIR License) licensed, as found in the LICENSE file.
