import utils

# From Paper Repository, symlinked from submodule (needs to be pulled and link needs to exist)
from idsds.models.resnet import resnet18, resnet50, resnet101, resnet152, wide_resnet50_2
from idsds.models.vgg import vgg16, vgg16_bn, vgg13, vgg19, vgg11
from idsds.models.ViT.ViT_new import vit_base_patch16_224
from idsds.models.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
from idsds.models.bagnets.pytorchnet import bagnet33
from idsds.models.xdnns.xfixup_resnet import xfixup_resnet50, fixup_resnet50
from idsds.models.xdnns.xvgg import xvgg16
from idsds.models.bcos_v2.bcos_resnet import resnet50 as bcos_resnet50
from idsds.models.bcos_v2.bcos_resnet import resnet18 as bcos_resnet18

original_models = "/workspace/hd/original/"
tuned_models = "/workspace/hd/tuned/"

resnet18_ood = resnet18(pretrained=True)
resnet18_id = resnet18(pretrained=True)
resnet18_id = utils.load_state_dict(
    tuned_models + "resnet18_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    resnet18_id
)

resnet50_ood = resnet50(pretrained=True)
resnet50_id = resnet50(pretrained=True)
resnet50_id = utils.load_state_dict(
    tuned_models + "resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    resnet50_id
)

resnet101_ood = resnet101(pretrained=True)
resnet101_id = resnet101(pretrained=True)
resnet101_id = utils.load_state_dict(
    tuned_models + "resnet101_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    resnet101_id
)

resnet152_ood = resnet152(pretrained=True)
resnet152_id = resnet152(pretrained=True)
resnet152_id = utils.load_state_dict(
    tuned_models + "resnet152_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    resnet152_id
)

wide_resnet50_ood = wide_resnet50_2(pretrained=True)
wide_resnet50_id = wide_resnet50_2(pretrained=True)
wide_resnet50_id = utils.load_state_dict(
    tuned_models + "wide_resnet50_2_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    wide_resnet50_id
)

vgg11_ood = vgg11(pretrained=True)
vgg11_id = vgg11(pretrained=True)
vgg11_id = utils.load_state_dict(
    tuned_models + "vgg11_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    vgg11_id
)

vgg13_ood = vgg13(pretrained=True)
vgg13_id = vgg13(pretrained=True)
vgg13_id = utils.load_state_dict(
    tuned_models + "vgg13_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    vgg13_id
)

vgg16_ood = vgg16(pretrained=True)
vgg16_id = vgg16(pretrained=True)
vgg16_id = utils.load_state_dict(
    tuned_models + "vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    vgg16_id
)

vgg16_bn_ood = vgg16_bn(pretrained=True)
vgg16_bn_id = vgg16_bn(pretrained=True)
vgg16_bn_id = utils.load_state_dict(
    tuned_models + "vgg16_bn_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    vgg16_bn_id
)

vgg19_ood = vgg19(pretrained=True)
vgg19_id = vgg19(pretrained=True)
vgg19_id = utils.load_state_dict(
    tuned_models + "vgg19_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    vgg19_id
)

fixup_resnet50_ood = fixup_resnet50()
fixup_resnet50_id = fixup_resnet50()
fixup_resnet50_id = utils.load_state_dict(
    tuned_models + "fixup_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    fixup_resnet50_id
)

x_resnet50_ood = xfixup_resnet50()
x_resnet50_id = xfixup_resnet50()
x_resnet50_id = utils.load_state_dict(
    tuned_models + "xresnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    x_resnet50_id
)

x_vgg16_ood = xvgg16(pretrained=True)
x_vgg16_id = xvgg16(pretrained=True)
x_vgg16_id = utils.load_state_dict(
    tuned_models + "xvgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    x_vgg16_id
)

bagnet33_ood = bagnet33(pretrained=True)
bagnet33_id = bagnet33(pretrained=True)
bagnet33_id = utils.load_state_dict(
    tuned_models + "bagnet33_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
    bagnet33_id
)

def get_all_pairs():
    return {
        "Resnet-50 / VGG-16": (resnet50_ood, vgg16_ood),
        "Resnet-18": (resnet18_ood, resnet18_id),
        "Resnet-50": (resnet50_ood, resnet50_id),
        "Resnet-101": (resnet101_ood, resnet101_id),
        "Resnet-152": (resnet152_ood, resnet152_id),
        "Wide Resnet-50": (wide_resnet50_ood, wide_resnet50_id),
        "Fixup Resnet-50": (fixup_resnet50_ood, fixup_resnet50_id),
        "X Resnet-50": (x_resnet50_ood, x_resnet50_id),
        "VGG-11": (vgg11_ood, vgg11_id),
        "VGG-13": (vgg13_ood, vgg13_id),
        "VGG-16": (vgg16_ood, vgg16_id),
        "VGG-19": (vgg19_ood, vgg19_id),
        "BN VGG-16": (vgg16_bn_ood, vgg16_bn_id),
        "X VGG-16": (x_vgg16_ood, x_vgg16_id),
        "Bagnet-33": (bagnet33_ood, bagnet33_id),
    }

def get_softmax_test_pairs():
    return {
        "Resnet-50 / VGG-16": (resnet50_ood, vgg16_ood),
        "Resnet-18": (resnet18_ood, resnet18_id),
        "Resnet-50": (resnet50_ood, resnet50_id),
        "Resnet-101": (resnet101_ood, resnet101_id),
        "Resnet-152": (resnet152_ood, resnet152_id),
        "Wide Resnet-50": (wide_resnet50_ood, wide_resnet50_id),
        "Fixup Resnet-50": (fixup_resnet50_ood, fixup_resnet50_id),
        "X Resnet-50": (x_resnet50_ood, x_resnet50_id),
        "VGG-11": (vgg11_ood, vgg11_id),
        "VGG-13": (vgg13_ood, vgg13_id),
        "VGG-16": (vgg16_ood, vgg16_id),
        "VGG-19": (vgg19_ood, vgg19_id),
        "BN VGG-16": (vgg16_bn_ood, vgg16_bn_id),
        "X VGG-16": (x_vgg16_ood, x_vgg16_id),
        #"Bagnet-33": (bagnet33_ood, bagnet33_id),
    }


if __name__ == "__main__":
    print(get_all_pairs().keys())