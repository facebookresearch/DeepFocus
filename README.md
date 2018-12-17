## DeepFocus

This repository provides source code, network models and dataset (~17GB) of the DeepFocus project from Facebook Reality Labs.

The source code and network models were implemented with TensorFlow and 32-bit precision. To further improve the runtime performance of the method, we recommond use of NVIDIA TensorRT toolbox to optimize the network inferences with 16-bit precision on your test machines. In general, we observed very minor quality decrease (less than 0.01dB in PSNR) while about 8 times speed up after the inference optimization. 

If you use our code and/or dataset, please cite our paper: 
Lei Xiao, Anton Kaplanyan, Alexander Fix, Matt Chapman, Douglas Lanman, "DeepFocus: Learned Image Synthesis For Computational Displays", SIGGRAPH Asia 2018 Technical Paper.

The technical paper, video and more supplementary materials can be found at: 
https://research.fb.com/publications/deepfocus-siggraph-asia-2018/

## License
DeepFocus is CC-BY-NC 4.0 (FAIR License) licensed, as found in the LICENSE file.
