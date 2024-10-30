# [CVPR 2024] Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion
### [Paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Yi_Text-IF_Leveraging_Semantic_Text_Guidance_for_Degradation-Aware_and_Interactive_Image_CVPR_2024_paper.pdf) | [Arxiv](https://arxiv.org/pdf/2403.16387.pdf) | [Code](https://github.com/XunpengYi/Text-IF) 

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

Recently, researchers have actively carried out research on enhanced image fusion involving degradation perception. We construct a fusion dataset benchmark with multiple degradation types on exist dataset, which implicates multiple explicit degradations on the sources images. 

The dataset is now available in [here](https://github.com/XunpengYi/EMS). Thanks to the MFNet, RoadScene/FLIR_aligned, LLVIP contributors.

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
⭐⭐ News: We released a dataset containing more diverse degradations (EMS-Full) for the degradation-robust image fusion task. All manual restored data and simulated degradation code are available. For more details, please refer to [EMS](https://github.com/XunpengYi/EMS) dataset.

## 3. Pretrained Weights
The pretrained weights for general image fusion performance is at [Google Drive](https://drive.google.com/file/d/1B_xOgS3SYIiTRGX2sZJ-0Vckbc76yXDQ/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1iMtAUaetvSS4Upf8BWIE8w) (code: nh9y).

The pretrained weights for text guidance image fusion performance is at [Google Drive](https://drive.google.com/file/d/13yYzMUV_EXu-y2Yb82rDy0MmzPcGqGCY/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1VdedSkychRRxkUP-fbRiYw) (code: cgrm).


## 4. Testing
For general image fusion performance comparison, please do not input the text with degradation prompt to ensure relative fairness.
```shell
# MFNet
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/MFNet/eval" --input_text "This is the infrared and visible light image fusion task." --save_path "./results"

# RoadScene
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/RoadScene/eval" --input_text "This is the infrared and visible light image fusion task."  --save_path "./results"

# LLVIP
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/simple_fusion.pth" --dataset_path "./dataset/LLVIP/eval" --input_text "This is the infrared and visible light image fusion task."  --save_path "./results"
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

### News: Text-IF supports handling more degradation (all types in EMS) and more powerful text prompts. 
The model weights for more degradation have now been made publicly available in [Google Drive](https://drive.google.com/file/d/1jstLiOp-ZBppz_vZhyG55YeUYFLyOdrP/view?usp=sharing) | [Baidu Drive](https://pan.baidu.com/s/1wydIlgurftN5tyCa6Nt7mg) (code: rwth). 

We only recommend using this weight when handling a large number of degradations. In general, we recommend using the previously mentioned at `Pretrained Weights` for text guidance image fusion.
```shell
# vis Low light
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "In the context of infrared-visible image fusion, visible images are susceptible to extremely low light degradation." --save_path "./results"

# vis Overexposure
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "We're tackling the infrared-visible image fusion challenge, dealing with visible images suffering from overexposure degradation." --save_path "./results"

# vis Random noise
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "The goal is to effectively fuse infrared and visible light images, mitigating the random noise present in visible images." --save_path "./results"

# vis Haze
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "We're tackling the fusion of infrared and visible light images, specifically focusing on haze issues in the visible images." --save_path "./results"

# vis Rain
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "This is the infrared-visible light fusion task, where visible images are affected by rain degradation." --save_path "./results"

# vis Blur
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "This task involves integrating infrared and visible light images, focusing on the degradation caused by blur in visible images." --save_path "./results"

# ir Low contrast
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "In this challenge, we're addressing the fusion of infrared and visible images, with a specific focus on the low contrast degradation in the infrared images." --save_path "./results"

# ir Stripe noise
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "This pertains to the fusion of infrared and visible light images, with an emphasis on addressing stripe noise degradation in the infrared images." --save_path "./results"

# ir Random noise
CUDA_VISIBLE_DEVICES=0 python test_from_dataset.py  --weights_path "pretrained_weights/text_fusion_power.pth" --dataset_path "./dataset/your_dataset/eval" --input_text "We're working on the fusion of infrared and visible images, with special consideration for the noise degradation affecting the infrared captures." --save_path "./results"
```
### Gallery

From left to right are the infrared image, visible image, and the fusion image obtained by Text-IF with the text guidance.

![Gallery](assert/LLVIP__010007.png)
![Gallery](assert/FLIR_RS__01932.png)

## 5. Train
Please prepare the training data as required from the EMS dataset. 
```bash
    dataset/
        dataset/
            train/
              # The infrared and visible images are low quality by default.
              type_degradation_1/
                  Infrared/
                  Infrared_gt/
                  Visible/
                  Visible_gt/
                  text/
              type_degradation_2/
                  Infrared/
                  Infrared_gt/
                  Visible/
                  Visible_gt/
                  text/
              type_degradation_3/
                  ...
            eval/
                Infrared/
                Visible/
                # You can enter the text while running the code, the text folder is optional here.
```
Modify the text prompt path in the `scripts/utils.py` to the corresponding configuration file and the dataset path in the `train_fusion.py`.
After that, run the following command:
```shell
python train_fusion.py
```

## Citation
If you find our work or dataset useful for your research, please cite our paper. 
```
@inproceedings{yi2024text,
  title={Text-IF: Leveraging Semantic Text Guidance for Degradation-Aware and Interactive Image Fusion},
  author={Yi, Xunpeng and Xu, Han and Zhang, Hao and Tang, Linfeng and Ma, Jiayi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024}
}
```
If you use the dataset of another work, please cite them as well and follow their licence. Here, we express our thanks to them. 
If you have any questions, please send an email to xpyi2008@163.com. 