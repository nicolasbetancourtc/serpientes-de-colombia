import os
import torch
import time
import requests
import pandas as pd
from pathlib import Path
from io import BytesIO
from PIL import Image
from itertools import chain
from torchvision import datasets
from .utils import taxon_metadata_request

def get_device(preference: str) -> str:
    """
    Return the device type as a string, honoring user preference if available.
    Falls back to 'cpu' if not available.
    """
    # Normalize input
    preference = preference.lower()

    # Get list of available backends
    available_devices = []

    if torch.cuda.is_available():
        available_devices.append("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():  # macOS GPU
        available_devices.append("mps")
    # You could add other checks here, like for "xpu", if you're using Intel's PyTorch extensions

    # Always fallback to cpu
    available_devices.append("cpu")

    # If user-preferred device is available, return it
    if preference in available_devices:
        return preference
    else:
        print(f"[Warning] Requested device '{preference}' not available. Using 'cpu' instead.")
        return "cpu"

def get_taxon_metadata(families_of_interest,genus_of_interest):
    families_data=taxon_metadata_request(families_of_interest,Taxa='family')
    genus_data=taxon_metadata_request(genus_of_interest,Taxa='genus')
    taxon_metadata=pd.concat([families_data,genus_data]).reset_index(drop=True)
    return taxon_metadata

def get_image_urls(taxon_metadata,INAT_URL,desired_observations,max_results_per_page, place_id ):
    dataframes=[]
    for idx, row in taxon_metadata.iterrows():
        params = {
            "taxon_id": row['id'] ,
            "place_id": place_id,
            "per_page": max_results_per_page,
            "photos": True,
            "page":1,
            "quality_grade":'research'
        }
        results = requests.get(INAT_URL, params=params)
        results.raise_for_status()
        data = results.json()
        urls=list(chain.from_iterable([ [ photo.get('url') for photo in obs.get('photos')] for obs in data['results']]))
        #Photo size
        urls = [u.replace("square", "original") for u in urls]
        desired_observations=min(desired_observations,data.get('total_results'))
        pages=desired_observations//max_results_per_page+1
        if pages>1:
            for page in range(2,pages+1):
                params['page']=page
                results = requests.get(INAT_URL, params=params)
                results.raise_for_status()
                data = results.json()
                urls+=list(chain.from_iterable([ [ photo.get('url') for photo in obs.get('photos')] for obs in data['results']]))
        taxa_df=pd.DataFrame({'url':urls, 
                              'id':   row['id'   ] ,		
                              'rank': row['rank' ] ,		
                              'name': row['name' ] ,		
                              'label':row['label'],
                             })
        dataframes.append(taxa_df)
    image_urls=pd.concat(dataframes)
    return image_urls

def download_images(image_urls, delay=0.2):
    image_urls=image_urls.sample(frac=1)
    image_urls=image_urls[image_urls.groupby('name')['label'].transform('rank', method='first')<=10].reset_index(drop=True)
    for _, row in image_urls.iterrows():
        
        try:
            r = requests.get(row['url'], timeout=10)
            img = Image.open(BytesIO(r.content))
            path = Path(row['label']) / f"{_}_{row['name']}"
            print('path',path)
            yield {path.as_posix() :img}
        except Exception as e:
            print(f"Failed to download {row['url']}: {e}")

    if delay > 0:
        time.sleep(delay)  # be nice to the server



    