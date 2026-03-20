from torchvision import datasets
import pandas as pd
import numpy as np

def filter_failed_downloads(serpientes_de_colombia_images,
                             unfiltered_image_urls
    ):
    succesfful_downloads=[  file_path.split('/')[1]+'.jpg' for  file_path, method in serpientes_de_colombia_images.items()]
    return unfiltered_image_urls[unfiltered_image_urls['file_name'].isin(succesfful_downloads)]

