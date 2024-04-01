from PIL import ImageFilter, Image

degradation = {
    "blur" : lambda *args, **kwargs: ImageFilter.BLUR(*args, **kwargs),
    "Gaussian Blur": lambda *args, **kwargs :  ImageFilter.GaussianBlur(*args, **kwargs),
    "Box Blur": lambda *args, **kwargs:  ImageFilter.BoxBlur(*args, **kwargs),
    "Median Filter" : lambda *args, **kwargs: ImageFilter.MedianFilter(*args, **kwargs),
    "Unsharp" : lambda *args, **kwargs : ImageFilter.MedianFilter(*args, **kwargs)
}

def degrade(effect_name : str, image : Image, **kwargs) -> Image:
    if effect_name not in degradation: return image
    return degradation(**kwargs)