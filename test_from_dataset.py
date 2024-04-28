import os
import numpy as np
from PIL import Image
import cv2
import clip
import torch
from torchvision.transforms import functional as F
from model.Text_IF_model import Text_IF as create_model
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main(args):
    root_path = args.dataset_path
    save_path = args.save_path
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF']
    text_line = args.input_text

    visible_root = os.path.join(root_path, "Visible")
    infrared_root = os.path.join(root_path, "Infrared")

    visible_path = [os.path.join(visible_root, i) for i in os.listdir(visible_root)
                  if os.path.splitext(i)[-1] in supported]
    infrared_path = [os.path.join(infrared_root, i) for i in os.listdir(infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    visible_path.sort()
    infrared_path.sort()

    print("Find the number of visible image: {},  the number of the infrared image: {}".format(len(visible_path), len(infrared_path)))
    assert len(visible_path) == len(infrared_path), "The number of the source images does not match!"

    print("Begin to run!")
    with torch.no_grad():
        model_clip, _ = clip.load("ViT-B/32", device=device)
        model = create_model(model_clip).to(device)

        model_weight_path = args.weights_path
        model.load_state_dict(torch.load(model_weight_path, map_location=device)['model'])
        model.eval()

    for i in range(len(visible_path)):
        ir_path = infrared_path[i]
        vi_path = visible_path[i]

        img_name = vi_path.replace("\\", "/").split("/")[-1]
        assert os.path.exists(ir_path), "file: '{}' dose not exist.".format(ir_path)
        assert os.path.exists(vi_path), "file: '{}' dose not exist.".format(vi_path)

        ir = Image.open(ir_path).convert(mode="RGB")
        vi = Image.open(vi_path).convert(mode="RGB")

        height, width = vi.size
        new_width = (width // 16) * 16
        new_height = (height // 16) * 16

        ir = ir.resize((new_height, new_width))
        vi = vi.resize((new_height, new_width))

        ir = F.to_tensor(ir)
        vi = F.to_tensor(vi)

        ir = ir.unsqueeze(0).cuda()
        vi = vi.unsqueeze(0).cuda()
        with torch.no_grad():
            text = clip.tokenize(text_line).to(device)
            i = model(vi, ir, text)
            fused_img_Y = tensor2numpy(i)
            save_pic(fused_img_Y, save_path, img_name)

        print("Save the {}".format(img_name))
    print("Finish! The results are saved in {}.".format(save_path))

def tensor2numpy(img_tensor):
    img = img_tensor.squeeze(0).cpu().detach().numpy()
    img = np.transpose(img, [1, 2, 0])
    return img

def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1.squeeze(0).detach().cpu().numpy()
    Y_channel = np.transpose(Y_channel, [1, 2, 0])
    img2 = img2.squeeze(0).cpu().numpy()
    img2 = np.transpose(img2, [1, 2, 0])
    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel, CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)
    return merged_img

def save_pic(outputpic, path, index : str):
    outputpic[outputpic > 1.] = 1
    outputpic[outputpic < 0.] = 0
    outputpic = cv2.UMat(outputpic).get()
    outputpic = cv2.normalize(outputpic, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
    outputpic=outputpic[:, :, ::-1]
    save_path = os.path.join(path, index).replace(".jpg", ".png")
    cv2.imwrite(save_path, outputpic)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='test data root path')
    parser.add_argument('--weights_path', type=str, required=True, help='initial weights path')
    parser.add_argument('--save_path', type=str, default='./results', help='output save image path')
    parser.add_argument('--input_text', type=str, required=True, help='text control input')

    parser.add_argument('--device', default='cuda', help='device (i.e. cuda or cpu)')
    parser.add_argument('--gpu_id', default='0', help='device id (i.e. 0, 1, 2 or 3)')
    opt = parser.parse_args()
    main(opt)