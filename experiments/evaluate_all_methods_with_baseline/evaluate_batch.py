# code based on https://github.com/visinf/idsds/blob/main/evaluate.py

import os
import argparse
import random
import torch
import gc
from tqdm import tqdm

import traceback
import multiprocessing as mp
from multiprocessing import Process, Queue

from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, GuidedBackprop, InputXGradient, Saliency, DeepLift, NoiseTunnel
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM

from utils.log import AverageMeter, ProgressMeter, Summary, accuracy, save_checkpoint
from models.model_wrapper import StandardModel, ViTModel, BcosModel
from models.resnet import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2
from models.vgg import vgg16, vgg16_bn, vgg13, vgg19, vgg11
from models.ViT.ViT_new import vit_base_patch16_224
from models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from models.bagnets.pytorchnet import bagnet33
from models.xdnns.xfixup_resnet import xfixup_resnet50, fixup_resnet50
from models.xdnns.xvgg import xvgg16
from models.bcos_v2.bcos_resnet import resnet50 as bcos_resnet50
from models.bcos_v2.bcos_resnet import resnet18 as bcos_resnet18

from utils.utils_batch import get_imagenet_loaders, str2bool
from explainers.explainer_wrapper import CaptumAttributionExplainer, CaptumNoiseTunnelAttributionExplainer, TorchcamExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, BcosExplainer, BagNetExplainer, RiseExplainer, DummyAttributionExplainer, EdgeDetectionExplainer, FrequencyExplainer
from single_deletion import single_deletion_protocol
from incremental_deletion import incremental_deletion_protocol


model_weights_dict = {
    'resnet18' : 'resnet18_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'resnet50' : 'resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'resnet101' : 'resnet101_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'resnet152' : 'resnet152_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vgg11' : 'vgg11_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vgg13' : 'vgg13_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vgg16' : 'vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vgg19' : 'vgg19_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vgg16_bn' : 'vgg16_bn_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'bagnet33' : 'bagnet33_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'wide_resnet50_2' : 'wide_resnet50_2_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'fixup_resnet50' : 'fixup_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'x_resnet50' : 'xresnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'vit_base_patch16_224' : 'vit_base_patch16_224_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'bcos_resnet50' : 'bcos_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar',
    'x_vgg16' : 'xvgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar'
    # missing:
    # 'bagnet9'
    # 'bcos_resnet18'
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data_dir', metavar='DIR', default='imagenet',
                    help='path to dataset (default: imagenet)')
# parser.add_argument('--model', required=True,
#                     choices=['resnet18', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'fixup_resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn', 'bagnet33', 'x_resnet50', 'vit_base_patch16_224', 'bcos_resnet50', 'x_vgg16'],
#                     help='model architecture')
# parser.add_argument('--explainer', required=True,
#                     choices=['Gradient', 'IxG', 'IG', 'IG-U', 'IG-SG', 'IxG-SG', 'IG-SG-SQ', 'IG-SG-VG', 'EG', 'AGI', 'Grad-CAM', 'Grad-CAMpp', 'SG-CAMpp', 'XG-CAM', 'Layer-CAM', 'Score-CAM', 'SS-CAM', 'IS-CAM', 'Rollout', 'CheferLRP', 'Bcos', 'BagNet', 'RISE', 'RISE-U', 'Dummy-Gaussian', 'Dummy-Entropy', 'Dummy-Random'],
#                     help='explainer')
parser.add_argument('--evaluation_protocol', required=True,
                    choices=['accuracy', 'single_deletion', 'incremental_deletion', 'accuracy_train_test_w_wo_patches'],
                    help='evaluation protocol to run')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch_size', default=64, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('--pretrained', default='False', type=str2bool,
#                     help='use pre-trained model')
# parser.add_argument('--pretrained_ckpt', type=str, default='none')
parser.add_argument('--pretrained_ckpt_path', type=str, required=True)
parser.add_argument('--output_file', type=str, required=True)
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use. If None, all GPUs are used')
# parser.add_argument('--overwrite', required=False, help='Overwrite result if already exists. If False, evaluation is skipped. Default=False', default='False', type=str2bool)


# parser.add_argument('--attribution_transform',
#                     choices=['raw', 'abs', 'relu'], default = 'raw',
#                     help='transformation applied to attribution')
parser.add_argument('--nr_images', default=-1, type=int,
                    help='number of images to use in the protocol. -1 for all images')

parser.add_argument('--use_softmax', default='False', type=str2bool,
                    help='compute attribution for each permutation new')

# for single_deletion
parser.add_argument('--grid_rows_and_cols', default=4, type=int,
                    help='number of rows and cols in the intervention grid')
parser.add_argument('--sd_baseline', required=False, default='zeros',
                    choices=['zeros', 'blur', 'average', 'random'],
                    help='baseline for perturbation')

# for incremental_deletion (id)
parser.add_argument('--id_baseline', required=False, default='zeros',
                    choices=['zeros', 'blur', 'average', 'random'],
                    help='baseline for perturbation')
parser.add_argument('--id_baseline_gaussian_kernel', default=51, type=int,
                    help='kernel size for Gaussian baseline')
parser.add_argument('--id_baseline_gaussian_sigma', default=41, type=int,
                    help='sigma for Gaussian baseline')
parser.add_argument('--id_steps', default=32, type=int,
                    help='number of steps for the protocol')
parser.add_argument('--id_order', required=False,
                    choices=['ascending', 'descending'],
                    default='ascending',
                    help='selection mode for perturbation')
parser.add_argument('--id_update_attribution', default='False', type=str2bool,
                    help='compute attribution for each permutation new')



def create_model(device, model_string, pretrained, pretrained_ckpt, explainer_string, use_softmax):
    if model_string == 'resnet50':
        model = resnet50(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4', use_softmax=use_softmax)
    elif model_string == 'resnet18':
        model = resnet18(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'resnet101':
        model = resnet101(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'resnet152':
        model = resnet152(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'wide_resnet50_2':
        model = wide_resnet50_2(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'fixup_resnet50':
        model = fixup_resnet50()
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'vgg11':
        model = vgg11(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif model_string == 'vgg13':
        model = vgg13(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif model_string == 'vgg16':
        model = vgg16(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features', use_softmax=use_softmax)
    elif model_string == 'vgg19':
        model = vgg19(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif model_string == 'vgg16_bn':
        model = vgg16_bn(pretrained=pretrained)
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif model_string == 'x_vgg16':
        model = xvgg16()
        model = StandardModel(model, gradcam_target_layer = 'model.features')
    elif model_string == 'bagnet33':
        model = bagnet33(pretrained=pretrained)
        model = StandardModel(model)
    elif model_string == 'x_resnet50':
        model = xfixup_resnet50()
        model = StandardModel(model, gradcam_target_layer = 'model.layer4')
    elif model_string == 'bcos_resnet50':
        model = bcos_resnet50(pretrained=pretrained, long_version=False)
        model = BcosModel(model)
    elif model_string == 'bcos_resnet18':
        model = bcos_resnet18(pretrained=pretrained)
        model = BcosModel(model)

    elif model_string == 'vit_base_patch16_224':
        if explainer_string == 'CheferLRP':
            model = vit_LRP(pretrained=pretrained)
        else:
            model = vit_base_patch16_224(pretrained=pretrained)
        model = ViTModel(model)
    else:
        print('Model not implemented')

    if pretrained_ckpt != 'none':
        state_dict = torch.load(pretrained_ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in state_dict.keys():
            state_dict = state_dict['state_dict']

        new_state_dict = state_dict
        if 'module.model.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("module.model."):
                    name = k[13:] # remove `model.`
                else:
                    name = k
                new_state_dict[name] = v
        elif 'module.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
        elif 'model.' in list(state_dict.keys())[0]:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("model."):
                    name = k[6:] # remove `model.`
                else:
                    name = k
                new_state_dict[name] = v

        model.load_state_dict(new_state_dict)

    model = model.to(device)
    model.eval()

    return model

def create_explainer(device, model, model_string, explainer_string, attribution_transform, seed, use_softmax, sigma=None, kernel_size=None):
    if explainer_string == 'IxG':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=attribution_transform)
    elif explainer_string == 'Gradient':
        explainer = Saliency(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=attribution_transform)
    elif explainer_string == 'IG':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=attribution_transform)
    elif explainer_string == 'IG-U':
        baseline = torch.rand((1,3,224,224)).to(device) * 2. - 1. # range is -1 to 1 which is approximately image range
        explainer = IntegratedGradients(model)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad', attribution_transform=attribution_transform)
    elif explainer_string == 'IxG-SG':
        explainer = InputXGradient(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, nt_type='smoothgrad', attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG-SQ':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad_sq', attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG-VG':
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='vargrad', attribution_transform=attribution_transform)
    elif explainer_string == 'Grad-CAM':
        if model_string != 'vit_base_patch16_224':
            explainer = GradCAM(model, target_layer=model.gradcam_target_layer)
            explainer = TorchcamExplainer(explainer, model)
        elif model_string == 'vit_base_patch16_224':
            explainer = ViTGradCamExplainer(model)
    elif explainer_string == 'Grad-CAMpp':
        explainer = GradCAMpp(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'SG-CAMpp':
        explainer = SmoothGradCAMpp(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'XG-CAM':
        explainer = XGradCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'Layer-CAM':
        explainer = LayerCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'Score-CAM':
        explainer = ScoreCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'SS-CAM':
        explainer = SSCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'IS-CAM':
        explainer = ISCAM(model, target_layer=model.gradcam_target_layer)
        explainer = TorchcamExplainer(explainer, model)
    elif explainer_string == 'Rollout':
        explainer = ViTRolloutExplainer(model)
    elif explainer_string == 'CheferLRP':
        explainer = ViTCheferLRPExplainer(model)
    elif explainer_string == 'Bcos':
        explainer = BcosExplainer(model)
    elif explainer_string == 'BagNet':
        explainer = BagNetExplainer(model)
    elif explainer_string == 'RISE':
        assert use_softmax == False # make sure the model does not use softmax output because it is used in RISE
        baseline = torch.zeros((1,3,224,224)).to(device)
        explainer = RiseExplainer(model, seed, baseline)
    elif explainer_string == 'RISE-U':
        assert use_softmax == False # make sure the model does not use softmax output because it is used in RISE
        baseline = torch.rand((1,3,224,224)).to(device) * 2. - 1. # range is -1 to 1 which is approximately image range
        explainer = RiseExplainer(model, seed, baseline)
    elif explainer_string == 'Dummy-Random':
        explainer = DummyAttributionExplainer('random')
    elif explainer_string == 'Dummy-Random-Squared':
        explainer = DummyAttributionExplainer('random-squared')
    elif explainer_string == 'Dummy-Random-Plus-One':
        explainer = DummyAttributionExplainer('random-plus-one')
    elif explainer_string == 'Dummy-Gaussian':
        assert sigma is not None
        explainer = DummyAttributionExplainer('gaussian', sigma=sigma)
    elif explainer_string == 'Dummy-Entropy':
        assert kernel_size is not None
        explainer = DummyAttributionExplainer('entropy', kernel_size=kernel_size)
    elif explainer_string == 'Edge-Sobel':
        explainer = EdgeDetectionExplainer('sobel')
    elif explainer_string == 'Edge-Gradient':
        explainer = EdgeDetectionExplainer('gradient')
    elif explainer_string == 'Edge-Canny':
        explainer = EdgeDetectionExplainer('canny')
    elif explainer_string == 'Edge-Marr-Hildreth':
        explainer = EdgeDetectionExplainer('marr-hildreth')
    elif explainer_string == 'Frequency-low':
        explainer = FrequencyExplainer('low')
    elif explainer_string == 'Frequency-high':
        explainer = FrequencyExplainer('high')
    elif explainer_string == 'Frequency-band':
        explainer = FrequencyExplainer('band')
    else:
        print('Explainer not implemented')

    return explainer

def evaluate_implementation(queue, args, seed, data_dir, batch_size, workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax, sigma=None, kernel_size=None):
    try:
        random.seed(seed)
        torch.manual_seed(seed)

        train_loader, val_loader = get_imagenet_loaders(data_dir, model_string, batch_size, workers, shuffle_val=True, train_with_eval_transform=True)

        model = create_model(device, model_string, pretrained, pretrained_ckpt, explainer_string, use_softmax)

        explainer = create_explainer(device, model, model_string, explainer_string, attribution_transform, seed, use_softmax, sigma=sigma, kernel_size=kernel_size)

        result = single_deletion_protocol(model, explainer, val_loader, args, device)
        queue.put(result)
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(("error", f"{e}\n{tb}"))

    # to avoid torch.OutOfMemoryError
    # del train_loader, val_loader, model, explainer
    # gc.collect()
    # torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    #return result

def evaluate(args, seed, data_dir, batch_size, workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax, sigma=None, kernel_size=None):
    queue = Queue()

    p = Process(target=evaluate_implementation, args=(queue, args, seed, data_dir, batch_size, workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax, sigma, kernel_size))

    p.start()
    p.join()


    if not queue.empty():
        result = queue.get()
    else:
        raise RuntimeError("Subprocess failed or didn't return a result.")

    del queue, p
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return result

class SimpleFileWriter:
    def __init__(self, path):
        self.f = open(path, "a", buffering=1)

    def write(self, msg):
        if not msg.endswith("\n"):
            msg += "\n"
        self.f.write(msg)
        self.f.flush()

    def close(self):
        self.f.close()

# def get_batch_size(args_batch_size, model_string, explainer_string):
#     batch_size = args_batch_size
#
#     if model_string in ['resnet50', 'x_resnet50', 'bcos_resnet50']:
#         batch_size = batch_size // 2
#     if model_string in ['resnet101', 'resnet152']:
#         batch_size = batch_size // 4
#
#     if explainer_string in ['IG', 'IG-U']:
#         batch_size = batch_size // 2
#     if explainer_string == 'IG-SG':
#         batch_size = batch_size // 4
#
#     return max(1, batch_size)

def main():
    args = parser.parse_args()

    writer = SimpleFileWriter(args.output_file)

    pretrained = False # otherwise the code will attempt to download and that will result in exception on the GPU service

    if args.gpu:
        device = 'cuda:' + str(args.gpu)
    else:
        device = 'cuda'

    mp.set_start_method("spawn", force=True)

    for model_string in ['resnet18', 'resnet50', 'resnet101', 'resnet152', 'wide_resnet50_2', 'fixup_resnet50', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 'vgg16_bn', 'x_resnet50', 'x_vgg16']:

        pretrained_ckpt = f'{args.pretrained_ckpt_path}/{model_weights_dict[model_string]}'

        for explainer_string in ['Gradient', 'IxG', 'IxG-SG', 'IG', 'IG-U', 'IG-SG']:
            for attribution_transform in ['raw', 'abs']:
                result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
                writer.write(f'{model_string},{explainer_string}-{attribution_transform},{result}')

        explainer_string = 'IG-SG-SQ'
        attribution_transform = 'abs'
        result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
        writer.write(f'{model_string},{explainer_string}-{attribution_transform},{result}')

        attribution_transform = 'raw'
        for explainer_string in ['Grad-CAM', 'Grad-CAMpp', 'SG-CAMpp', 'XG-CAM', 'Layer-CAM']:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
            writer.write(f'{model_string},{explainer_string},{result}')

        explainer_string='Dummy-Gaussian'
        for sigma in [1, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 128, 160, 256]:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax, sigma=sigma)
            writer.write(f'{model_string},{explainer_string}-{sigma},{result}')

        explainer_string='Dummy-Entropy'
        for kernel_size in [3, 5, 7, 9, 13, 17, 31]:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax, kernel_size=kernel_size)
            writer.write(f'{model_string},{explainer_string}-{kernel_size},{result}')

        for explainer_string in ['Dummy-Random', 'Dummy-Random-Squared', 'Dummy-Random-Plus-One']:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
            writer.write(f'{model_string},{explainer_string},{result}')

        for explainer_string in ['Edge-Sobel', 'Edge-Gradient', 'Edge-Canny', 'Edge-Marr-Hildreth', 'Frequency-low', 'Frequency-band', 'Frequency-high']:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_string, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
            writer.write(f'{model_string},{explainer_string},{result}')

    writer.close()

if __name__ == '__main__':
    main()
