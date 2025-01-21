from torchvision import transforms
from transformers import AutoImageProcessor

train_transforms = transforms.Compose(
    [
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transforms = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

def preprocess_with_processor():
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
    return transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
