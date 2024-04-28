# [CVPR 2024] Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion
### Paper | [Arxiv](https://arxiv.org/pdf/2403.16387.pdf) | [Code](https://github.com/XunpengYi/Text-IF)
[![arXiv](https://img.shields.io/badge/arxiv-paper-179bd3)](https://arxiv.org/pdf/2403.16387.pdf) 

**Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion**
Xunpeng Yi, Han Xu, Hao Zhang, Linfeng Tang and Jiayi Ma in CVPR 2024


![Framework](assert/framework.png)

## 1. Create Environment
- Create Conda Environment
```
conda create -n textif_env python=3.8
conda activate textif_env
```
- Install Dependencies
```
conda install pytorch==2.0.0 torchvision==0.15.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```
If you have the device with the lower version of CUDA, you can try to install a lower version of torch, but torch > 1.10 is recommended. Please refer to the official website of PyTorch.

## 2. Prepare Your Dataset

EMS Dataset: Enhanced Multi-Spectral Various Scenarios for Degradation-Aware Image Fusion

Recently, researchers have actively carried out research on enhanced image fusion involving degradation perception. We construct a fusion dataset benchmark with multiple degradation types on exist dataset, which implicates multiple explicit degradations on the sources images. We are now applied for the licence.
The link of the full dataset (EMS-Full) will be available in June expectedly.

You can also refer to [MFNet](https://www.mi.t.utokyo.ac.jp/static/projects/mil_multispectral/), [RoadScene](https://github.com/hanna-xu/RoadScene)/[FLIR_aligned](https://adas-dataset-v2.flirconservator.com/#downloadguide), [LLVIP](https://github.com/bupt-ai-cz/LLVIP) to prepare your data. You should list your dataset as followed rule:
```bash
    dataset/
        your_dataset/
            train/
                Infrared/
                Visible/
            eval/
                Infrared/
                Visible/
```

## 3. Pretrain Weights
The pretrain weights for general image fusion performance is at [Google Drive](https://drive.google.com/file/d/146jH_-6oquoEKc1HnMoWxLcu9mwjWACF/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1wrDfkocRE4aa_mX-PQG_kA) (code: nh9x).

The pretrain weights for text guidance image fusion performance is at [Google Drive](https://drive.google.com/file/d/1p4Isv-lTqIMpY4io_jB8fFa696mAq_XF/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/11gsYyxQOjhzSX0rJ9SA2-w) (code: cgrv).

## 4. Testing
For general image fusion performance comparison, please do not input the text with degradation prompt to ensure relative fairness.
```shell
# MFNet
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/MFNet/eval" --input_text "This is the infrared and visible image fusion task." --save_path "./results"

# RoadScene
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/RoadScene/eval" --input_text "This is the infrared and visible image fusion task."  --save_path "./results"

# LLVIP
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/LLVIP/eval" --input_text "This is the infrared and visible image fusion task."  --save_path "./results"
```

For text guidance image fusion, the existing model supports hints for low light, overexposure, low contrast, and noise. Feel free to use it.
```shell
# Low light
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "In the context of infrared-visible image fusion, visible images are susceptible to extremely low light degradation." --save_path "./results"

# Overexposure
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "We're tackling the infrared-visible image fusion challenge, dealing with visible images suffering from overexposure degradation." --save_path "./results"

# Low contrast
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "In this challenge, we're addressing the fusion of infrared and visible images, with a specific focus on the low contrast degradation in the infrared images." --save_path "./results"

# Noise
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "We're working on the fusion of infrared and visible images, with special consideration for the noise degradation affecting the infrared captures." --save_path "./results"
```
Stay tuned for more powerful model support with more tasks and descriptions in the future.

### Gallery

From left to right are the infrared image, visible image, and the fusion image obtained by Text-IF with the text guidance.

![Gallery](assert/LLVIP__010007.png)
![Gallery](assert/FLIR_RS__01932.png)

## 5. Train
The training code will be released with the EMS-Full dataset in June for research purposes only. 

## Citation
If you find our work or dataset useful for your research, please cite our paper. 
```
@article{yi2024text,
  title={Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion},
  author={Yi, Xunpeng and Xu, Han and Zhang, Hao and Tang, Linfeng and Ma, Jiayi},
  journal={arXiv preprint arXiv:2403.16387},
  year={2024}
}

@inproceedings{yi2024text,
  title={Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion},
  author={Yi, Xunpeng and Xu, Han and Zhang, Hao and Tang, Linfeng and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
If you use the dataset of another work, please cite them as well and follow their licence. Here, we express our thanks to them. 
If you have any questions, please send an email to xpyi2008@163.com. 