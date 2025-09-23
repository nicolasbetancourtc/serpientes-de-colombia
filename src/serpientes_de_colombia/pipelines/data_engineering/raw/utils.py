import requests
import pandas as pd
def url_request(family=None, genus=None):
    if family:
        rank='family'
        return f"https://api.inaturalist.org/v1/taxa?q={family}&is_active=true&rank={rank}&order=desc&order_by=observations_count"
    elif genus:
        rank='genus'
        return f"https://api.inaturalist.org/v1/taxa?q={genus}&is_active=true&rank={rank}&order=desc&order_by=observations_count"
def taxon_metadata_request(taxa_of_interest,Taxa):
    pd_dfs=[]
    for label, taxa_list in taxa_of_interest.items():
        urls=[url_request(**{Taxa:taxa}) for taxa in taxa_list]
        json_results=[requests.get(url).json() for url in urls]
        dict_results=[result['results'][0] 
                      for result in json_results
                      if len(result['results'])>0
                     ]
        pd_df=pd.DataFrame([{**result,**{'label':label}} for result in dict_results])
        pd_dfs.append(pd_df)
    prefered_columns=['id','rank','name','label']
    df=pd.concat(pd_dfs)
    return df[prefered_columns]

def get_image_urls(
    taxon_id,
    place_id=7196, 
    max_links=500,
    research_only=True
):
    urls = []
    per_page = 200       # API maximum
    page = 1

    while len(urls) < max_links:
        params = {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "per_page": per_page,
            "page": page,
            "photos": True,
        }
        if research_only:
            params["quality_grade"] = "research"

        r = requests.get(INAT_URL, params=params, headers=HEADERS)
        r.raise_for_status()
        data = r.json()

        results = data.get("results", [])
        if not results:
            break  # no more data

        for obs in results:
            for p in obs.get("photos", []):
                u = p.get("url")
                if not u:
                    continue
                if "square" in u:  # upgrade to higher res
                    u = u.replace("square", "original")
                urls.append(u)
                if len(urls) >= max_links:
                    break
            if len(urls) >= max_links:
                break

        page += 1

    return urls