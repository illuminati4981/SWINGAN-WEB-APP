from PIL import ImageFilter, Image

degradation = {
    "blur" : lambda : ImageFilter.BLUR,
    "Gaussian Blur": lambda radius :  ImageFilter.GaussianBlur(radius=radius),
    "Box Blur": lambda radius :  ImageFilter.BoxBlur(radius=radius),
    "Median Filter" : lambda size: ImageFilter.MedianFilter(size=size),
    "Unsharp" : lambda radius, percent, threshold : ImageFilter.MedianFilter(radius, percent, threshold)
}

def degrade(effect_name : str, image : Image, **kwargs) -> Image:
    if effect_name not in degradation: return image
    return degradation(**kwargs)