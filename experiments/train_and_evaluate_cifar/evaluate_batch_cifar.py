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

from models.model_wrapper import StandardModel

from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, GuidedBackprop, InputXGradient, Saliency, DeepLift, NoiseTunnel
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM

from utils.log import AverageMeter, ProgressMeter, Summary, accuracy, save_checkpoint
from utils.cifar_utils import get_cifar100_loaders, str2bool
from explainers.explainer_wrapper import CaptumAttributionExplainer, CaptumNoiseTunnelAttributionExplainer, TorchcamExplainer, ViTGradCamExplainer, ViTRolloutExplainer, ViTCheferLRPExplainer, BcosExplainer, BagNetExplainer, RiseExplainer
from single_deletion import single_deletion_protocol
from incremental_deletion import incremental_deletion_protocol

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


def get_last_conv_layer(model):
    for layer in reversed(model.features):
        if isinstance(layer, torch.nn.Conv2d):
            return layer
    raise ValueError("No Conv2d layer found in model.features")

def create_model(device, model_name, pretrained, pretrained_ckpt, use_softmax):
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=pretrained)

    # Identify the target layer for GradCAM
    if "resnet" in model_name:
        if hasattr(model, 'layer4'):
            target_layer = 'model.layer4'
        elif hasattr(model, 'layer3'):
            target_layer = 'model.layer3'
        else:
            raise ValueError(f"Cannot determine GradCAM target layer for model: {model_name}")
        model = StandardModel(model, gradcam_target_layer=target_layer, use_softmax=use_softmax)

    elif "vgg" in model_name:
        target_layer = get_last_conv_layer(model)
        model = StandardModel(model, gradcam_target_layer=target_layer, use_softmax=use_softmax)

    else:
        raise NotImplementedError(f"Model not implemented: {model_name}")


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

def create_explainer(device, model, model_name, explainer_string, attribution_transform, seed, use_softmax):
    if explainer_string == 'IxG':
        explainer = InputXGradient(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=attribution_transform)
    elif explainer_string == 'Gradient':
        explainer = Saliency(model)
        explainer = CaptumAttributionExplainer(explainer, attribution_transform=attribution_transform)
    elif explainer_string == 'IG':
        explainer = IntegratedGradients(model)
        baseline = torch.zeros((1,3,32,32)).to(device)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=attribution_transform)
    elif explainer_string == 'IG-U':
        baseline = torch.rand((1,3,32,32)).to(device) * 2. - 1. # range is -1 to 1 which is approximately image range
        explainer = IntegratedGradients(model)
        explainer = CaptumAttributionExplainer(explainer, baseline=baseline, attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG':
        baseline = torch.zeros((1,3,32,32)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad', attribution_transform=attribution_transform)
    elif explainer_string == 'IxG-SG':
        explainer = InputXGradient(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, nt_type='smoothgrad', attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG-SQ':
        baseline = torch.zeros((1,3,32,32)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='smoothgrad_sq', attribution_transform=attribution_transform)
    elif explainer_string == 'IG-SG-VG':
        baseline = torch.zeros((1,3,32,32)).to(device)
        explainer = IntegratedGradients(model)
        explainer = NoiseTunnel(explainer)
        explainer = CaptumNoiseTunnelAttributionExplainer(explainer, baseline=baseline, nt_type='vargrad', attribution_transform=attribution_transform)
    elif explainer_string == 'Grad-CAM':
        if model_name != 'vit_base_patch16_224':
            explainer = GradCAM(model, target_layer=model.gradcam_target_layer)
            explainer = TorchcamExplainer(explainer, model)
        elif model_name == 'vit_base_patch16_224':
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
    else:
        print('Explainer not implemented')

    return explainer

def evaluate_implementation(queue, args, seed, data_dir, batch_size, workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax):
    try:
        random.seed(seed)
        torch.manual_seed(seed)

        train_loader, val_loader = get_cifar100_loaders(args, shuffle_val=True, train_with_eval_transform=True)

        model = create_model(device, model_name, pretrained, pretrained_ckpt, use_softmax)

        explainer = create_explainer(device, model, model_name, explainer_string, attribution_transform, seed, use_softmax)

        result = single_deletion_protocol(model, explainer, val_loader, args, device)
        queue.put(result)
    except Exception as e:
        tb = traceback.format_exc()
        queue.put(("error", f"{e}\n{tb}"))

def evaluate(args, seed, data_dir, batch_size, workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax):
    queue = Queue()

    p = Process(target=evaluate_implementation, args=(queue, args, seed, data_dir, batch_size, workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, use_softmax))

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

def main():
    args = parser.parse_args()

    writer = SimpleFileWriter(args.output_file)

    pretrained = False # otherwise the code will attempt to download and that will result in exception on the GPU service

    if args.gpu:
        device = 'cuda:' + str(args.gpu)
    else:
        device = 'cuda'

    mp.set_start_method("spawn", force=True)

    for model_name in ['cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44', 'cifar100_resnet56', 'cifar100_vgg11_bn', 'cifar100_vgg13_bn', 'cifar100_vgg16_bn', 'cifar100_vgg19_bn']:


        pretrained_ckpt = f'{args.pretrained_ckpt_path}/{model_name}_checkpoint_best.pth.tar'

        for explainer_string in ['Gradient', 'IxG', 'IxG-SG', 'IG', 'IG-U', 'IG-SG']:
            for attribution_transform in ['raw', 'abs']:
                result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
                writer.write(f'{model_name},{explainer_string}-{attribution_transform},{result}')

        explainer_string = 'IG-SG-SQ'
        attribution_transform = 'abs'
        result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
        writer.write(f'{model_name},{explainer_string}-{attribution_transform},{result}')

        attribution_transform = 'raw'
        for explainer_string in ['Grad-CAM', 'Grad-CAMpp', 'SG-CAMpp', 'XG-CAM', 'Layer-CAM']:
            result = evaluate(args, args.seed, args.data_dir, args.batch_size, args.workers, device, model_name, explainer_string, attribution_transform, pretrained, pretrained_ckpt, args.use_softmax)
            writer.write(f'{model_name},{explainer_string},{result}')

    writer.close()

if __name__ == '__main__':
    main()
