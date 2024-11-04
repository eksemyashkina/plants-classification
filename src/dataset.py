from typing import List, Dict, Tuple, Callable
from pathlib import Path
from torch.utils.data import Dataset
import PIL.Image
import torch


class PlantsDataset(Dataset):
    def __init__(
        self,
        root: str,
        labels: Dict[int, str],
        transform: Callable,
        load_to_ram: bool = True,
    ) -> None:
        super().__init__()
        self.root = root
        self.labels = labels
        self.transform = transform
        self.load_to_ram = load_to_ram

        self.data = [
            {
                "path": x.as_posix(),
                "label": self.labels[x.parent.name],
                "image": PIL.Image.open(x).convert("RGB") if self.load_to_ram else None,
            }
            for x in sorted(Path(self.root).glob("**/*.jpg"))
        ]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.data[idx]
        if self.load_to_ram:
            image = item["image"]
        else:
            image = PIL.Image.open(item["path"]).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(item["label"], dtype=torch.long)
        return (image, label)


def collate_fn(items: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    images = torch.cat([item[0].unsqueeze(0) for item in items])
    labels = torch.cat([item[1].unsqueeze(0) for item in items])
    return (images, labels)
