import torch

model_list = ['cifar100_resnet20', 'cifar100_resnet32', 'cifar100_resnet44', 'cifar100_resnet56', 'cifar100_vgg11_bn', 'cifar100_vgg13_bn', 'cifar100_vgg16_bn', 'cifar100_vgg19_bn']


for model_name in model_list:
    model = torch.hub.load("chenyaofo/pytorch-cifar-models", model_name, pretrained=True)
    model_path = f"{model_name}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
