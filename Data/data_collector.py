from bs4 import BeautifulSoup
import requests
import shutil


def download_file(url):
    local_filename = url.split('/')[-1]
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192): 
                # If you have chunk encoded response uncomment if
                # and set chunk_size parameter to None.
                #if chunk: 
                f.write(chunk)
    return local_filename

html_text = requests.get('https://www.flyingmag.com/photo-gallery-photos-top-100-airplanes/').text

soup = BeautifulSoup(html_text, 'lxml')

planeFigs = soup.find_all('img')
planeImg = []

class_attribute = 'wp-block-image image-block-'
i = 0
for planeFig in planeFigs:
    try:
        download_file(planeFig.attrs['data-lazy-src'])
    except:
        print("")
    i += 1    

    





