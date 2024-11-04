import torchvision.transforms as T

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = T.Compose([
    T.RandomRotation(degrees=15),
    T.RandomResizedCrop(224, scale=(0.5, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

test_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=mean, std=std),
])

class EMA:
    def __init__(self, alpha: float = 0.9) -> None:
        self.value = None
        self.alpha = alpha
    
    def __call__(self, value: float) -> float:
        if self.value is None:
            self.value = value
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * value
        return self.value
