import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from captum.attr import LayerAttribution
from torchvision import transforms
from PIL import Image

class AbstractExplainer():
    def __init__(self, explainer, attribution_transform='raw', baseline = None):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer).__name__
        self.baseline = baseline
        self.attribution_transform = attribution_transform

    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)


class AbstractAttributionExplainer(AbstractExplainer):
    
    @abstractmethod
    def explain(self, input):
        return self.explainer.explain(self.model, input)
    
class CaptumAttributionExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def explain(self, input, target=None, baseline=None):
        if self.explainer_name == 'Saliency':
            attr = self.explainer.attribute(input, target=target, abs=False)
        elif self.explainer_name == 'InputXGradient': 
            attr = self.explainer.attribute(input, target=target) 
        elif self.explainer_name == 'LayerGradCam': 
            B,C,H,W = input.shape
            attr = self.explainer.attribute(input, target=target, relu_attributions=True)
            m = transforms.Resize((H,W), interpolation=Image.NEAREST)
            attr = m(attr)

        elif self.explainer_name == 'IntegratedGradients':
            attr = self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50)

        attr = attr.sum(dim=1, keepdim=True)

        if self.attribution_transform == 'raw':
            attr = attr
        elif self.attribution_transform == 'abs':
            attr = attr.abs()
        elif self.attribution_transform == 'relu':
            m = nn.ReLU()
            attr = m(attr)

        return attr

        
class CaptumNoiseTunnelAttributionExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def __init__(self, explainer, baseline = None, nt_type='smoothgrad', attribution_transform='raw'):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.explainer_name = type(self.explainer.attribution_method).__name__
        self.baseline = baseline
        self.nt_type = nt_type
        self.nt_samples = 4
        self.attribution_transform = attribution_transform


    def explain(self, input, target=None, baseline=None):
        
        if self.explainer_name == 'Saliency' or self.explainer_name == 'InputXGradient': 
            attr= self.explainer.attribute(input, target=target, nt_type=self.nt_type, nt_samples=self.nt_samples)
            
        elif self.explainer_name == 'IntegratedGradients':
            attr = self.explainer.attribute(input, target=target, baselines=self.baseline, n_steps=50, nt_type=self.nt_type, nt_samples=self.nt_samples)
            
        attr = attr.sum(dim=1, keepdim=True)

        if self.attribution_transform == 'raw':
            attr = attr
        elif self.attribution_transform == 'abs':
            attr = attr.abs()
        elif self.attribution_transform == 'relu':
            m = nn.ReLU()
            attr = m(attr)

        return attr
    
class TorchcamExplainer(AbstractAttributionExplainer):
    
    """
    A wrapper for Captum attribution methods.
    Args:
        explainer: Captum explanation method
    """
    def __init__(self, explainer, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = explainer
        self.model = model
        #self.explainer_name = type(self.explainer).__name__
        self.resizer = transforms.Resize((224,224), interpolation=Image.BILINEAR)
        #self.baseline = baseline
        #self.nt_type = nt_type
        #self.nt_samples = 10
        #print(self.explainer_name)


    def explain(self, input, target=None, baseline=None):
        B,C,H,W = input.shape
        cams_for_batch = []
        for b_idx in range(B):
            out = self.model(input[b_idx].unsqueeze(0))
            cams = self.explainer(target[b_idx].item(), out)
            assert len(cams) == 1
            cam = cams[0]#.unsqueeze(0).unsqueeze(0)
            #B, C, H, W = cam.shape
            #assert C == 1
            #cam = cam[0,0,:,:] # remove channel for resizing
            cam = self.resizer(cam)
            cam = cam.unsqueeze(0)
            #print(cam.shape)
            cams_for_batch.append(cam)
        return torch.cat(cams_for_batch, dim = 0)

class DummyAttributionExplainer(AbstractAttributionExplainer):
    def __init__(self, kind='random', sigma=0.5, kernel_size=5):
        self.kind = kind
        self.resizer = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)
        self.sigma = sigma
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size


    def explain(self, input, target=None, baseline=None):
        B, C, H, W = input.shape
        assert C == 3, "DummyAttributionExplainer expects 3-channel RGB input"
        attr_batch = []

        for b_idx in range(B):
            if self.kind == 'random':
                attr = torch.rand((1, H, W))  # single-channel saliency
            elif self.kind == 'random-squared':
                attr = torch.rand((1, H, W))
            elif self.kind == 'random-plus-one':
                attr = torch.rand((1, H, W))+1
            elif self.kind == 'gaussian':
                x = torch.linspace(-1, 1, W)
                y = torch.linspace(-1, 1, H)
                grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
                sigma = self.sigma
                gaussian = torch.exp(-(grid_x**2 + grid_y**2) / (2 * sigma**2))
                attr = gaussian.unsqueeze(0)  # shape: [1, H, W]
            elif self.kind == 'entropy':
                img = input[b_idx]
                gray = img.mean(dim=0, keepdim=True)  # [1, H, W]
                kernel_size = self.kernel_size
                padding = (kernel_size - 1) // 2
                # Using average pooling on squared local deviations from the mean is a very rough proxy for local entropy or local complexity.
                # It’s not true entropy (which involves probability distributions and log terms), but it tries to cheaply mimic areas with more variation in pixel values.
                entropy_map = torch.nn.functional.avg_pool2d(
                    (gray - gray.mean())**2, kernel_size=kernel_size, stride=1, padding=padding
                )
                attr = entropy_map  # shape: [1, H, W]
            else:
                raise ValueError(f"Unknown dummy explainer kind: {self.kind}")

            # Resize and expand to 3 channels
            attr_resized = self.resizer(attr)  # shape: [1, 224, 224]
            attr_3ch = attr_resized.expand(3, -1, -1)  # shape: [3, 224, 224]
            attr_batch.append(attr_3ch.unsqueeze(0))  # shape: [1, 3, 224, 224]

        return torch.cat(attr_batch, dim=0)  # shape: [B, 3, 224, 224]


import torch.nn.functional as F
import cv2

class EdgeDetectionExplainer(AbstractAttributionExplainer):
    def __init__(self, method="sobel"):
        assert method in {"sobel", "gradient", "canny", "marr-hildreth"}, f"Unsupported method: {method}"
        self.method = method
        self.resizer = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

    def _to_gray(self, img: torch.Tensor) -> torch.Tensor:
        return img.mean(dim=0, keepdim=True)  # [1, H, W]

    def _sobel(self, gray: torch.Tensor) -> torch.Tensor:
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], device=gray.device).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [ 0,  0,  0],
                                [ 1,  2,  1]], device=gray.device).float().unsqueeze(0).unsqueeze(0)
        dx = F.conv2d(gray.unsqueeze(0), sobel_x, padding=1).squeeze(0)  # remove batch dimension, now [1, H, W]
        dy = F.conv2d(gray.unsqueeze(0), sobel_y, padding=1).squeeze(0)  # remove batch dimension, now [1, H, W]
        return dx, dy

    def _laplacian(self, gray: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor([[0,  1, 0],
                               [1, -4, 1],
                               [0,  1, 0]], device=gray.device).float().unsqueeze(0).unsqueeze(0)
        return F.conv2d(gray.unsqueeze(0), kernel, padding=1).squeeze(0) # remove batch dimension, now [1, H, W]

    def _gaussian_blur(self, gray: torch.Tensor, kernel_size=5, sigma=1.0) -> torch.Tensor:
        return transforms.GaussianBlur(kernel_size, sigma)(gray)

    def _canny_cv2(self, img: torch.Tensor) -> torch.Tensor:
        img_np = img.permute(1, 2, 0).cpu().detach().numpy() * 255
        img_np = img_np.astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        edge_tensor = torch.tensor(edges / 255.0, dtype=torch.float32).unsqueeze(0)
        return edge_tensor.to(img.device)

    def explain(self, input: torch.Tensor, target=None, baseline=None) -> torch.Tensor:
        B, C, H, W = input.shape
        edge_batch = []

        for b_idx in range(B):
            img = input[b_idx]
            gray = self._to_gray(img)

            if self.method == "sobel":
                dx, dy = self._sobel(gray)
                edge = dx.abs() + dy.abs()

            elif self.method == "gradient":
                dx, dy = self._sobel(gray)
                edge = torch.sqrt(dx**2 + dy**2 + 1e-8)

            elif self.method == "marr-hildreth":
                blurred = self._gaussian_blur(gray)
                edge = self._laplacian(blurred)

            elif self.method == "canny":
                edge = self._canny_cv2(img)

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Resize and expand to 3 channels
            edge_resized = self.resizer(edge)  # shape: [1, 224, 224]
            edge_3ch = edge_resized.expand(3, -1, -1)  # shape: [3, 224, 224]
            edge_batch.append(edge_3ch.unsqueeze(0))  # shape: [1, 3, 224, 224]

        return torch.cat(edge_batch, dim=0)  # shape: [B, 3, 224, 224]

import torch.fft
class FrequencyExplainer(AbstractAttributionExplainer):
    def __init__(self, method="high"):
        assert method in {"high", "band", "low"}, f"Unsupported method: {method}"
        self.method = method
        self.resizer = transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR)

    def explain(self, input: torch.Tensor, target=None, baseline=None) -> torch.Tensor:
        B, C, H, W = input.shape
        freq_batch = []

        for b_idx in range(B):
            img = input[b_idx]

            # Convert to grayscale using luminosity method (if RGB)
            if img.shape[0] == 3:
                grayscale = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
            else:
                grayscale = img[0]

            # Apply FFT
            fft = torch.fft.fft2(grayscale)
            fft_shifted = torch.fft.fftshift(fft)

            # Create frequency mask
            y = torch.linspace(-1, 1, H, device=img.device).view(-1, 1).expand(H, W)
            x = torch.linspace(-1, 1, W, device=img.device).view(1, -1).expand(H, W)
            radius = torch.sqrt(x**2 + y**2)

            if self.method == "low":
                mask = (radius < 0.4).float()
            elif self.method == "high":
                mask = (radius > 0.6).float()
            elif self.method == "band":
                mask = ((radius > 0.3) & (radius < 0.6)).float()
            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Apply mask and reconstruct
            fft_filtered = fft_shifted * mask
            fft_unshifted = torch.fft.ifftshift(fft_filtered)
            filtered_img = torch.fft.ifft2(fft_unshifted).real

            # Normalize to [0, 1]
            filtered_img = (filtered_img - filtered_img.min()) / (filtered_img.max() - filtered_img.min() + 1e-8)

            # Reshape and format as 3-channel
            filtered_img = filtered_img.unsqueeze(0)  # [1, H, W]
            resized = self.resizer(filtered_img)      # [1, 224, 224]
            freq_3ch = resized.expand(3, -1, -1)       # [3, 224, 224]
            freq_batch.append(freq_3ch.unsqueeze(0))  # [1, 3, 224, 224]

        return torch.cat(freq_batch, dim=0)  # [B, 3, 224, 224]

        #     # Resize and expand to 3 channels
        #     freq_resized = self.resizer(normed)  # shape: [1, 224, 224]
        #     freq_3ch = freq_resized.expand(3, -1, -1)  # shape: [3, 224, 224]
        #     freq_batch.append(freq_3ch.unsqueeze(0))  # shape: [1, 3, 224, 224]
        #
        # return torch.cat(freq_batch, dim=0)  # shape: [B, 3, 224, 224]


from models.ViT.ViT_explanation_generator import Baselines, LRP
class ViTGradCamExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_cam_attn(input_, index=target).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

class ViTRolloutExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = Baselines(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_rollout(input_, start_layer=1).reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

class ViTCheferLRPExplainer(AbstractAttributionExplainer):
    def __init__(self, model):
        self.model = model
        self.explainer = LRP(self.model.model)

    def explain(self, input, target):
        B,C,H,W = input.shape
        assert B == 1
        input_ = torch.nn.functional.interpolate(input, (224,224))
        attribution = self.explainer.generate_LRP(input_, index=target, start_layer=1, method="transformer_attribution").reshape(1, 1, 14, 14)
        m = transforms.Resize((H,W), interpolation=Image.NEAREST)
        attribution = m(attribution)
        return attribution

def explanation_mode(model, active=True):
    for mod in model.modules():
        if hasattr(mod, "explanation_mode"):
            mod.explanation_mode(active)

class AddInverse(nn.Module):

    def __init__(self, dim=1):
        """
            Adds (1-in_tensor) as additional channels to its input via torch.cat().
            Can be used for images to give all spatial locations the same sum over the channels to reduce color bias.
        """
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor):
        out = torch.cat([in_tensor, 1-in_tensor], self.dim)
        return out

from torch.autograd import Variable
class BcosExplainer(AbstractAttributionExplainer):
    
    def __init__(self, model):
        """
        An explainer for bcos explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model

    def explain(self, input, target):
        
        #explanation_mode(self.model, True)
        #print(self.model.model.)
        #output = self.model(input) # if directly using self.model, the gradient computation does not work
        #target = output[0][target]
        #input.grad = None
        #target[0].backward(retain_graph=True)
        #w1 = input.grad
        #attribution = w1 * input
        
        assert input.shape[0] == 1 # batch size = 1
        model = self.model.model
        with model.explanation_mode():
        
            _input = Variable(AddInverse()(input), requires_grad=True)  # not sure if this should be here or rather in the model wrapper.      
            output = model(_input) # if directly using self.model it returns None
            #model.explanation_mode()
            target = output[0][target]
            _input.grad = None
            target[0].backward(retain_graph=True)
            w1 = _input.grad
            attribution = w1 * _input
            attribution = attribution.sum(dim=1, keepdim=True)

        #self.model.model.explanation_mode()

        return attribution
        
from models.bagnets.utils import generate_heatmap_pytorch
from models.bagnets.utils import generate_heatmap_pytorch2

class BagNetExplainer(AbstractAttributionExplainer):
    """
    A wrapper for LIME.
    https://github.com/marcotcr/lime/blob/master/doc/notebooks/Tutorial%20-%20images%20-%20Pytorch.ipynb
    Args:
        model: PyTorch model.
    """
    def __init__(self, model):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.model = model

    def explain(self, input, target):
        assert input.shape[0] == 1
        attribution_numpy = generate_heatmap_pytorch(self.model, input.cpu(), target, 33)
        attribution = torch.from_numpy(attribution_numpy).unsqueeze(0).unsqueeze(0).to(input.device)
        return attribution
    
from .rise import RISE
import os 
class RiseExplainer(AbstractAttributionExplainer):
    """
    A wrapper for RISE.
    Args:
        model: PyTorch model.
    """
    def __init__(self, model, seed, baseline):
        """
        An abstract wrapper for explanations.
        Args:
            model: PyTorch neural network model
        """
        self.explainer = RISE(model, (224, 224), baseline, gpu_batch=128)
        # Generate masks for RISE or use the saved ones.
        self.seed = seed
        maskspath = '/fastdata/rhesse/rise_masks_phd_imagenet_patches_evaluation_seed' + str(self.seed) + '.npy'
        generate_new = False

        if generate_new or not os.path.isfile(maskspath):
            self.explainer.generate_masks(N=1000, s=8, p1=0.1, savepath=maskspath)
            print('Masks are generated.')
        else:
            self.explainer.load_masks(maskspath, p1=0.1)
            print('Masks are loaded.')

    def explain(self, input, target=None):
        assert input.shape[0] == 1
        attribution = self.explainer(input)
        attribution = attribution[target[0].int()].unsqueeze(0).unsqueeze(0)
        
        return attribution
