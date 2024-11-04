from typing import Callable, List, Union
from pathlib import Path
import PIL.Image
import torch
from torch import nn
import torchvision.transforms.functional as F


class Conv2dNormActivation(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1,
        padding: int | None = None, groups: int = 1, norm_layer: Callable[..., torch.nn.Module] = nn.BatchNorm2d,
        activation_layer: Callable[..., torch.nn.Module] = nn.ReLU, bias: bool | None = False,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = norm_layer(out_channels)
        self.activation = activation_layer()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int,
    ) -> None:
        super().__init__()
        self.stride = stride
        
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            layers.append(Conv2dNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6))
        layers.extend(
            [
                Conv2dNormActivation(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    groups=hidden_dim,
                    norm_layer=nn.BatchNorm2d,
                    activation_layer=nn.ReLU6,
                ),
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        weights_path: str | None = None,
    ) -> None:
        super().__init__()

        if weights_path is not None and not Path(weights_path).exists():
            raise FileNotFoundError(weights_path)

        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = [Conv2dNormActivation(3, input_channel, stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6)]
        for t, c, n, s in inverted_residual_setting:
            output_channel = c
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        features.append(
            Conv2dNormActivation(
                input_channel, last_channel, kernel_size=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU6
            )
        )
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

        if weights_path:
            self.load_pretrained_weights(weights_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
    def load_pretrained_weights(self, weights_path: str) -> None:
        state_dict = torch.load(weights_path, map_location="cpu")
        model_state_dict = self.state_dict()
        new_state_dict = {}
        for key1, key2 in zip(model_state_dict.keys(), state_dict.keys()):
            new_state_dict[key1] = state_dict[key2]
        self.load_state_dict(new_state_dict)
    
    @torch.inference_mode()
    def predict(self, x: torch.Tensor, top_k: int | None) -> Union[List[int], List[List[int]]]:
        output = self.forward(x)
        probs = torch.nn.functional.softmax(output, dim=1)
        if top_k is not None:
            preds = torch.topk(probs, dim=1, k=top_k).indices
            return preds.tolist()
        else:
            pred = torch.argmax(probs, dim=1)
            return pred.tolist()


if __name__ == "__main__":
    model = MobileNetV2(weights_path="weights\mobilenet_v2-b0353104.pth")
    num_params = sum([p.numel() for p in model.parameters()])
    print(f"params: {num_params/1e6:.2f} M")

    model.eval()
    image = PIL.Image.open("assets\cat.jpg").convert("RGB")
    image = F.resize(image, (224, 224))
    image = F.to_tensor(image)
    image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image = image.unsqueeze(0)
    predicted_class = model.predict(image, top_k=10)
    print(f"predicted class: {predicted_class}")
    # https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/
