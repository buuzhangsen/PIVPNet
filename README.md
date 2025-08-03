# All-in-one Image Restoration Method via Prior Information and Visual Perception Prompt

<hr />

> **Abstract:** *All-in-one image restoration seeks to address diverse image degradations—such as haze, rain, noise, blur, and low-light conditions—within a single, unified network. However, a fundamental challenge lies in balancing high-fidelity restoration quality with strong generalization across heterogeneous degradation types, particularly under lightweight model constraints. In this work, we propose PIVPNet, a novel Prior-Informed and Visual-Prompted Network that introduces a dual-guidance adaptive learning framework for efficient multi-task image restoration. The core innovation of PIVPNet lies in its degradation-aware modular design, which synergistically integrates three key components: the Prior Information Guidance Module, which explicitly models degradation-specific priors to guide feature learning;  the Visual Perception Prompt Module, which introduces learnable visual prompts to dynamically adapt the network’s attention to varying degradation patterns; and the Lightweight Detail Enhancement Module, which jointly refines spatial and frequency-domain features to recover fine textures with minimal computational overhead. Unlike existing multi-task frameworks that rely on shared backbones with static fusion strategies, PIVPNet enables task-aware feature modulation through prompt-based conditional adaptation, significantly improving both restoration fidelity and model generalization. Numerous experiments have shown that PIVPNet can achieve good results with only one-third of the existing method parameters. Notably, on image dehazing, our method surpasses the best-performing model by 6\% in PSNR, and achieves an average gain of 0.16 dB across all tasks, despite its compact architecture. These results highlight the effectiveness of prior-informed guidance and prompt-driven perception adaptation in multi-task restoration, offering a new paradigm for efficient and generalizable all-in-one image restoration. * 
<hr />

## Network Architecture
<img src = "PIVPNet.jpg"> 

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py
```
to start the training of the model. Use the ```de_type``` argument to choose the combination of degradation types to train on. By default it is set to all the 5 degradation tasks (denoising, deraining, dehazing, deblurring, enhancement).

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py --de_type derain dehaze
```

## Testing

After preparing the testing data in ```test/``` directory, place the mode checkpoint file in the ```ckpt``` directory. To perform the evaluation, use
```
python test.py --mode {n}
```
```n``` is a number that can be used to set the tasks to be evaluated on, 0 for denoising, 1 for deraining, 2 for dehazing, 3 for deblurring, 4 for enhancement, 5 for three-degradation all-in-one setting and 6 for five-degradation all-in-one setting.

Example Usage: To test on all the degradation types at once, run:

```
python test.py --mode 6
```
<!-- 
## Demo
To obtain visual results from the model ```demo.py``` can be used. After placing the saved model file in ```ckpt``` directory, run:
```
python demo.py --test_path {path_to_degraded_images} --output_path {save_images_here}
```
Example usage to run inference on a directory of images:
```
python demo.py --test_path './test/demo/' --output_path './output/demo/'
```
Example usage to run inference on an image directly:
```
python demo.py --test_path './test/demo/image.png' --output_path './output/demo/'
```
To use tiling option while running ```demo.py``` set ```--tile``` option to ```True```. The Tile size and Tile overlap parameters can be adjusted using ```--tile_size``` and ```--tile_overlap``` options respectively. -->

## Results
