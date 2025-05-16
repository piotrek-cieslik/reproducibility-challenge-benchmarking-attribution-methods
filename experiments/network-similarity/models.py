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

class Models:
    def __init__(self, tuned_models):
        self.resnet18_ood = resnet18(pretrained=True)
        self.resnet18_id = resnet18(pretrained=True)
        self.resnet18_id = utils.load_state_dict(
            tuned_models + "/resnet18_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.resnet18_id
        )
        
        self.resnet50_ood = resnet50(pretrained=True)
        self.resnet50_id = resnet50(pretrained=True)
        self.resnet50_id = utils.load_state_dict(
            tuned_models + "/resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.resnet50_id
        )
        
        self.resnet101_ood = resnet101(pretrained=True)
        self.resnet101_id = resnet101(pretrained=True)
        self.resnet101_id = utils.load_state_dict(
            tuned_models + "/resnet101_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.resnet101_id
        )
        
        self.resnet152_ood = resnet152(pretrained=True)
        self.resnet152_id = resnet152(pretrained=True)
        self.resnet152_id = utils.load_state_dict(
            tuned_models + "/resnet152_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.resnet152_id
        )
        
        self.wide_resnet50_ood = wide_resnet50_2(pretrained=True)
        self.wide_resnet50_id = wide_resnet50_2(pretrained=True)
        self.wide_resnet50_id = utils.load_state_dict(
            tuned_models + "/wide_resnet50_2_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.wide_resnet50_id
        )
        
        self.vgg11_ood = vgg11(pretrained=True)
        self.vgg11_id = vgg11(pretrained=True)
        self.vgg11_id = utils.load_state_dict(
            tuned_models + "/vgg11_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.vgg11_id
        )
        
        self.vgg13_ood = vgg13(pretrained=True)
        self.vgg13_id = vgg13(pretrained=True)
        self.vgg13_id = utils.load_state_dict(
            tuned_models + "/vgg13_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.vgg13_id
        )
        
        self.vgg16_ood = vgg16(pretrained=True)
        self.vgg16_id = vgg16(pretrained=True)
        self.vgg16_id = utils.load_state_dict(
            tuned_models + "/vgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.vgg16_id
        )
        
        self.vgg16_bn_ood = vgg16_bn(pretrained=True)
        self.vgg16_bn_id = vgg16_bn(pretrained=True)
        self.vgg16_bn_id = utils.load_state_dict(
            tuned_models + "/vgg16_bn_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.vgg16_bn_id
        )
        
        self.vgg19_ood = vgg19(pretrained=True)
        self.vgg19_id = vgg19(pretrained=True)
        self.vgg19_id = utils.load_state_dict(
            tuned_models + "/vgg19_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.vgg19_id
        )
        
        self.fixup_resnet50_ood = fixup_resnet50()
        self.fixup_resnet50_id = fixup_resnet50()
        self.fixup_resnet50_id = utils.load_state_dict(
            tuned_models + "/fixup_resnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.fixup_resnet50_id
        )
        
        self.x_resnet50_ood = xfixup_resnet50()
        self.x_resnet50_id = xfixup_resnet50()
        self.x_resnet50_id = utils.load_state_dict(
            tuned_models + "/xresnet50_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.x_resnet50_id
        )
        
        self.x_vgg16_ood = xvgg16(pretrained=True)
        self.x_vgg16_id = xvgg16(pretrained=True)
        self.x_vgg16_id = utils.load_state_dict(
            tuned_models + "/xvgg16_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.x_vgg16_id
        )
        
        self.bagnet33_ood = bagnet33(pretrained=True)
        self.bagnet33_id = bagnet33(pretrained=True)
        self.bagnet33_id = utils.load_state_dict(
            tuned_models + "/bagnet33_imagenet1000_lr0.001_epochs30_step10_checkpoint_best.pth.tar", 
            self.bagnet33_id
        )

    def get_all_pairs(self):
        return {
            "Resnet-50 / VGG-16": (self.resnet50_ood, self.vgg16_ood),
            "Resnet-18": (self.resnet18_ood, self.resnet18_id),
            "Resnet-50": (self.resnet50_ood, self.resnet50_id),
            "Resnet-101": (self.resnet101_ood, self.resnet101_id),
            "Resnet-152": (self.resnet152_ood, self.resnet152_id),
            "Wide Resnet-50": (self.wide_resnet50_ood, self.wide_resnet50_id),
            "Fixup Resnet-50": (self.fixup_resnet50_ood, self.fixup_resnet50_id),
            "X Resnet-50": (self.x_resnet50_ood, self.x_resnet50_id),
            "VGG-11": (self.vgg11_ood, self.vgg11_id),
            "VGG-13": (self.vgg13_ood, self.vgg13_id),
            "VGG-16": (self.vgg16_ood, self.vgg16_id),
            "VGG-19": (self.vgg19_ood, self.vgg19_id),
            "BN VGG-16": (self.vgg16_bn_ood, self.vgg16_bn_id),
            "X VGG-16": (self.x_vgg16_ood, self.x_vgg16_id),
            "Bagnet-33": (self.bagnet33_ood, self.bagnet33_id),
        }
    
    def get_softmax_test_pairs(self):
        return {
            "Resnet-50 / VGG-16": (self.resnet50_ood, self.vgg16_ood),
            "Resnet-18": (self.resnet18_ood, self.resnet18_id),
            "Resnet-50": (self.resnet50_ood, self.resnet50_id),
            "Resnet-101": (self.resnet101_ood, self.resnet101_id),
            "Resnet-152": (self.resnet152_ood, self.resnet152_id),
            "Wide Resnet-50": (self.wide_resnet50_ood, self.wide_resnet50_id),
            "Fixup Resnet-50": (self.fixup_resnet50_ood, self.fixup_resnet50_id),
            "X Resnet-50": (self.x_resnet50_ood, self.x_resnet50_id),
            "VGG-11": (self.vgg11_ood, self.vgg11_id),
            "VGG-13": (self.vgg13_ood, self.vgg13_id),
            "VGG-16": (self.vgg16_ood, self.vgg16_id),
            "VGG-19": (self.vgg19_ood, self.vgg19_id),
            "BN VGG-16": (self.vgg16_bn_ood, self.vgg16_bn_id),
            "X VGG-16": (self.x_vgg16_ood, self.x_vgg16_id),
            #"Bagnet-33": (self.bagnet33_ood, self.bagnet33_id),
        }

    
    def get_attribution_test_dict_all(self):
        return {
            # "Model": (model1, model2, layer1, layer2)
            #"Resnet-50 / VGG-16": (self.resnet50_ood, self.vgg16_ood, "layer4", "features"),
            "Resnet-18": (self.resnet18_ood, self.resnet18_id, "layer4", "layer4"),
            "Resnet-50": (self.resnet50_ood, self.resnet50_id, "layer4", "layer4", ),
            "Resnet-101": (self.resnet101_ood, self.resnet101_id, "layer4", "layer4", ),
            "Resnet-152": (self.resnet152_ood, self.resnet152_id, "layer4", "layer4", ),
            "Wide Resnet-50": (self.wide_resnet50_ood, self.wide_resnet50_id, "layer4", "layer4", ),
            "Fixup Resnet-50": (self.fixup_resnet50_ood, self.fixup_resnet50_id, "layer4", "layer4", ),
            "X Resnet-50": (self.x_resnet50_ood, self.x_resnet50_id, "layer4", "layer4", ),
            "VGG-11": (self.vgg11_ood, self.vgg11_id, "features", "features"),
            "VGG-13": (self.vgg13_ood, self.vgg13_id, "features", "features"),
            "VGG-16": (self.vgg16_ood, self.vgg16_id, "features", "features"),
            "VGG-19": (self.vgg19_ood, self.vgg19_id, "features", "features"),
            "BN VGG-16": (self.vgg16_bn_ood, self.vgg16_bn_id, "features", "features"),
            "X VGG-16": (self.x_vgg16_ood, self.x_vgg16_id, "features", "features"),
            #"Bagnet-33": (bagnet33_ood, bagnet33_id),
        }
