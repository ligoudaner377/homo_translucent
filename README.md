# Inverse Scattering
This is the implementation for the CVPR 2023 paper:

"Inverse Rendering of Translucent Objects using Physical and Neural Renderers"

By [Chenhao Li](https://ligoudaner377.github.io/ "Chenhao Li"), [Trung Thanh Ngo](https://www.is.ids.osaka-u.ac.jp/author/trung-thanh-ngo/ "Trung Thanh Ngo"), [Hajime Nagahara](https://www.is.ids.osaka-u.ac.jp/author/hajime-nagahara/ "Hajime Nagahara")

[Project page](https://ligoudaner377.github.io/homo_translucent/) | [arXiv](https://arxiv.org/abs/2305.08336) | [Dataset](https://drive.google.com/file/d/150NljNZSuZ648Osy-hMizYYb10jJ44PC/view?usp=share_link) | [Video](https://www.youtube.com/watch?v=rWZLU_YqacE)

<div  align="center">    
<img src="Files/real.png" width="700">
</div>

## Requirements

* Linux
* NVIDIA GPU + CUDA CuDNN
* Python 3
* torch
* torchvision
* dominate
* visdom
* pandas
* scipy
* pillow

## How to use

### Test on the real-world objects
```bash
python ./inference_real.py --dataroot "./datasets/real" --dataset_mode "real" --name "edit_twoshot" --model "edit_twoshot" --eval
```

### Train
- Download the [dataset](https://drive.google.com/file/d/150NljNZSuZ648Osy-hMizYYb10jJ44PC/view?usp=share_link).
- Unzip it to ./datasets/
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train the model (change gpu_ids according your device)

```bash
python ./train.py --dataroot "./datasets/translucent" --name "edit_twoshot" --model "edit_twoshot" --gpu_ids 0,1,2,3
```

### Test
```bash
python ./test.py --dataroot "./datasets/translucent" --name "edit_twoshot" --model "edit_twoshot" --eval
```

### scripts.sh integrate all commands
```bash
bash ./scripts.sh
```
## Citation

```bash
@inproceedings{li2023inverse,
  title={Inverse Rendering of Translucent Objects using Physical and Neural Renderers},
  author={Li, Chenhao and Ngo, Trung Thanh and Nagahara, Hajime},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12510--12520},
  year={2023}}
```

## Acknowledgements

Code derived and modified from:

- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix")
- [Single-Shot Neural Relighting and SVBRDF Estimation](https://github.com/ssangx/NeuralRelighting "https://github.com/ssangx/NeuralRelighting")
