import torch
import torch.nn as nn
import timm
import torchvision.models as models
import time


class MapEmbedderSwinTiny(nn.Module):
    """
    Map encoding as context to condition the TrafficDiffuser using Swin Transformer Tiny.
    """
    def __init__(self, map_channels, hidden_size):
        super().__init__()

        # Load Swin Tiny model from timm and modify the first conv layer for 4-channel input
        self.swin = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
        self.swin.patch_embed.proj = nn.Conv2d(map_channels, 96, kernel_size=4, stride=4)
        self.swin.head = nn.Identity()
        swin_out_features = self.swin.num_features
        
        # Pooling after Swin Transformer 
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Final projection and normalization 
        self.norm_final = nn.LayerNorm(swin_out_features, elementwise_affine=False, eps=1e-6)
        self.proj_final = nn.Linear(swin_out_features, hidden_size, bias=True)

    def forward(self, x):
        # (B, C, H, W) -> (B, hidden_size)
        x = self.swin(x)                 # Output from Swin Tiny
        
        x = x.permute(0, 3, 1, 2)        # (B, 7, 7, swin_features) -> (B, swin_features, 7, 7)
        x = self.pool(x)                 # Pooling
        x = x.flatten(1)                 # Flatten
        
        x = self.norm_final(x)           # Apply LayerNorm
        x = self.proj_final(x)           # Project to hidden_size
        return x
    

class MapEmbedderConvNeXtTiny(nn.Module):
    """
    Map encoding using ConvNeXt-Tiny as context to condition the TrafficDiffuser.
    """ 
    def __init__(self, map_channels, hidden_size):
        super().__init__()      
        # Load pretrained ConvNeXt-Tiny
        self.convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        # Modify the first convolution layer to accept 4-channel input
        self.convnext.features[0][0] = nn.Conv2d(map_channels, 96, kernel_size=4, stride=4, padding=0, bias=False)
        # Remove the classifier layer at the end
        self.convnext = nn.Sequential(*list(self.convnext.children())[:-2])

        # Final projection and normalization
        self.norm_final = nn.LayerNorm(768, elementwise_affine=False, eps=1e-6)  # Adjusted for ConvNeXt output
        self.proj_final = nn.Linear(768, hidden_size, bias=True)

    def forward(self, x):
        # (B, C, H, W)
        x = self.convnext(x)            # (B, 768, H, W)
        x = x.mean(dim=[2, 3])          # Global average pooling (B, 768)
        x = self.norm_final(x)          # (B, 768)
        x = self.proj_final(x)          # (B, hidden_size)
        return x


class MapEmbedderEfficientNetB0(nn.Module):
    """
    Map encoding using EfficientNet-B0 as context to condition the TrafficDiffuser.
    """ 
    def __init__(self, map_channels, hidden_size):
        super().__init__()      
        # Load pretrained EfficientNet-B0
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        # Modify the first convolution layer to accept 4-channel input
        self.efficientnet.features[0][0] = nn.Conv2d(map_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
        # Remove the final fully connected layer
        self.efficientnet = nn.Sequential(*list(self.efficientnet.children())[:-2])
        
        # Final projection and normalization
        self.norm_final = nn.LayerNorm(1280, elementwise_affine=False, eps=1e-6)
        self.proj_final = nn.Linear(1280, hidden_size, bias=True)

    def forward(self, x):
        # (B, C, H, W)
        x = self.efficientnet(x)        # (B, 1280, H, W)
        x = x.mean(dim=[2, 3])          # Global average pooling (B, 1280)
        x = self.norm_final(x)          # (B, 1280)
        x = self.proj_final(x)          # (B, hidden_size)
        return x


# Define a function to test inference time
def test_inference_time(model, device, input_tensor, num_trials=100):
    model = model.to(device)
    input_tensor = input_tensor.to(device)
    
    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure time
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_trials):
            _ = model(input_tensor)
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / num_trials
    return avg_inference_time


# Test setup
map_channels = 4
hidden_size = 384
batch_size = 100
height = 224
width = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create random torch tensor for map input
random_map_tensor = torch.randn(batch_size, map_channels, height, width)

# Initialize the models
model_swin = MapEmbedderSwinTiny(map_channels, hidden_size)
model_convnext = MapEmbedderConvNeXtTiny(map_channels, hidden_size)
model_efficientnet = MapEmbedderEfficientNetB0(map_channels, hidden_size)

# Print the number of parameters
print(f"Swin Transformer Tiny number of parameters: {sum(p.numel() for p in model_swin.parameters() if p.requires_grad)}")
print(f"ConvNeXt-Tiny number of parameters: {sum(p.numel() for p in model_convnext.parameters() if p.requires_grad)}")
print(f"EfficientNet-B0 number of parameters: {sum(p.numel() for p in model_efficientnet.parameters() if p.requires_grad)}")

# Measure inference times
swin_time = test_inference_time(model_swin, device, random_map_tensor)
convnext_time = test_inference_time(model_convnext, device, random_map_tensor)
efficientnet_time = test_inference_time(model_efficientnet, device, random_map_tensor)

# Print inference times
print(f"Average inference time for Swin Transformer Tiny: {swin_time:.6f} seconds")
print(f"Average inference time for ConvNeXt-Tiny: {convnext_time:.6f} seconds")
print(f"Average inference time for EfficientNet-B0: {efficientnet_time:.6f} seconds")
