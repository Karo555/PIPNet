import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from features.resnet_features import resnet18_features, resnet34_features, resnet50_features, resnet50_features_inat, resnet101_features, resnet152_features
# from features.convnext_features import convnext_tiny_26_features, convnext_tiny_13_features 
import torch
from torch import Tensor
import sys
sys.path.append('../B-cos')
from modules.bcosconv2d import BcosConv2d

class PIPNet(nn.Module):
    def __init__(self,
                 num_classes: int,
                 num_prototypes: int,
                 feature_net: nn.Module,
                 args: argparse.Namespace,
                 add_on_layers: nn.Module,
                 classification_layer: nn.Module
                 ):
        super().__init__()
        assert num_classes > 0
        self._num_features = args.num_features
        self._num_classes = num_classes
        self._num_prototypes = num_prototypes
        self._net = feature_net
        self._add_on = add_on_layers
        self._classification = classification_layer
        self._multiplier = classification_layer.normalization_multiplier

    def forward(self, xs, inference=False):
        features = self._net(xs) 
        proto_features = self._add_on(features)
        
        # For B-cos networks, we work directly with proto_features
        # Apply global average pooling manually for classification
        pooled = torch.mean(proto_features.view(proto_features.size(0), proto_features.size(1), -1), dim=2)
        
        if inference:
            clamped_pooled = torch.where(pooled < 0.1, 0., pooled)  #during inference, ignore all prototypes that have 0.1 similarity or lower
            out = self._classification(clamped_pooled) #shape (bs, num_classes)
            return proto_features, clamped_pooled, out
        else:
            out = self._classification(pooled) #shape (bs, num_classes) 
            return proto_features, pooled, out


base_architecture_to_features = {'resnet18': resnet18_features,
                                 'resnet34': resnet34_features,
                                 'resnet50': resnet50_features,
                                 'resnet50_inat': resnet50_features_inat,
                                 'resnet101': resnet101_features,
                                 'resnet152': resnet152_features}
                                #  'convnext_tiny_26': convnext_tiny_26_features,
                                #  'convnext_tiny_13': convnext_tiny_13_features}

# B-cos linear transformation layer
class BcosLinear(nn.Module):
    """Applies a B-cos linear transformation to the incoming data using 1x1 convolution"""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super(BcosLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Use BcosConv2d with 1x1 kernel to implement linear transformation
        self.bcos_conv = BcosConv2d(in_features, out_features, kernel_size=1, stride=1, 
                                   padding=0, b=2, max_out=2)
        self.normalization_multiplier = nn.Parameter(torch.ones((1,), requires_grad=True))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, device=device, dtype=dtype))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: Tensor) -> Tensor:
        # Reshape input to add spatial dimensions for conv operation
        # input shape: (batch_size, in_features) -> (batch_size, in_features, 1, 1)
        input_conv = input.unsqueeze(-1).unsqueeze(-1)
        
        # Apply B-cos convolution
        output_conv = self.bcos_conv(input_conv)
        
        # Reshape back to linear format
        # output shape: (batch_size, out_features, 1, 1) -> (batch_size, out_features)
        output = output_conv.squeeze(-1).squeeze(-1)
        
        # Add bias if present
        if self.bias is not None:
            output = output + self.bias
            
        return output


def get_network(num_classes: int, args: argparse.Namespace): 
    features = base_architecture_to_features[args.net](pretrained=not args.disable_pretrained)
    features_name = str(features).upper()
    if 'next' in args.net:
        features_name = str(args.net).upper()
    if features_name.startswith('RES') or features_name.startswith('CONVNEXT'):
        # Look for BcosConv2d layers since we're using B-cos ResNet
        bcos_layers = [i for i in features.modules() if isinstance(i, BcosConv2d)]
        if bcos_layers:
            first_add_on_layer_in_channels = bcos_layers[-1].outc // bcos_layers[-1].max_out
        else:
            # Fallback to regular Conv2d if no BcosConv2d found
            conv_layers = [i for i in features.modules() if isinstance(i, nn.Conv2d)]
            first_add_on_layer_in_channels = conv_layers[-1].out_channels
    else:
        raise Exception('other base architecture NOT implemented')
    
    
    if args.num_features == 0:
        num_prototypes = first_add_on_layer_in_channels
        print("Number of prototypes: ", num_prototypes, flush=True)
        add_on_layers = nn.Sequential(
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    else:
        num_prototypes = args.num_features
        print("Number of prototypes set from", first_add_on_layer_in_channels, "to", num_prototypes,". Extra 1x1 conv layer added. Not recommended.", flush=True)
        add_on_layers = nn.Sequential(
            nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=num_prototypes, kernel_size=1, stride = 1, padding=0, bias=True), 
            nn.Softmax(dim=1), #softmax over every prototype for each patch, such that for every location in image, sum over prototypes is 1                
    )
    
    if args.bias:
        classification_layer = BcosLinear(num_prototypes, num_classes, bias=True)
    else:
        classification_layer = BcosLinear(num_prototypes, num_classes, bias=False)
        
    return features, add_on_layers, classification_layer, num_prototypes


    