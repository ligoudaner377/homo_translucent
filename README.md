# Inverse Rendering of Translucent Objects using Physical and Neural Renderers


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
python ./inference_real.py --dataroot "./datasets/real" --dataset_mode "real" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine" --eval
```
### Scene editing on the real-world objects
```bash
python ./scene_edit_real.py --dataroot "./datasets/real" --dataset_mode "real" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine" --isEdit True --eval
```

### Train
- Download the dataset. (not ready yet)
- Unzip it to ./datasets/
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Pretrain the model (change the gpu_ids based on your device)
```bash
python ./train.py --dataroot "./datasets/translucent" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine"  --step init --gpu_ids 0,1,2,3
```
- Refine the model (change the gpu_ids based on your device)
```bash
python ./train.py --dataroot "./datasets/translucent" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine"  --step refine --gpu_ids 0,1,2,3
```

### Test
```bash
python ./test.py --dataroot "./datasets/translucent" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine" --eval
```

### Scene editing on the synthetic data
```bash
python ./scene_edit.py --dataroot "./datasets/translucent" --name "twoshot_direct_refine15_2" --model "twoshotr_direct_refine" --isEdit True --eval
```

###scripts.sh integrate all the commands
```bash
bash ./scripts.sh
```

## Acknowledgements

Code derived and reshaped from:

- [pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix "https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix")
- [Single-Shot Neural Relighting and SVBRDF Estimation](https://github.com/ssangx/NeuralRelighting "https://github.com/ssangx/NeuralRelighting")
