from bs4 import BeautifulSoup
import requests
import shutil
import os


def download_file(url, name, num):
    folder_path = os.path.join('C:\\Users\\Aksha\\Documents\\GitHub\\ML\\Aircraft Pictures\\',name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    local_filename = folder_path + '\\' + str(num) + r'.jpg'
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

def whitespaceToSpace(raw):
    nameSearch = []
    i = 0
    for name in plane_raw:
        str = ""
        for i in range(len(name)):
            if (name[i] != " "):
                str += name[i]
            else:
                str += "+"
        nameSearch.append(str)
        i += 1
    return nameSearch

def photoCollection(name, i):
    search = "https://www.bing.com/images/search?q="+ name +"&form=HDRSC3&first="+ str(i) + "&tsc=ImageHoverTitle"  

    html_text = requests.get(search, stream=True).text

    soup = BeautifulSoup(html_text, 'lxml')

    return soup.find_all('img')

def photoLoop(name, planeFigs):
    i = 1
    print(len(planeFigs))
    for planeFig in planeFigs:
        try:
            download_file(planeFig.attrs['src'], name, i)
        except:
            continue

        i += 1

plane_raw = ["Boeing 787", 
"Airbus A380", 
"Airbus A300", 
"Airbus 350", 
"Airbus Beluga", 
"Airbus BelugaXL", 
"Boeing 777", 
"A-37A Dragonfly", 
"ADM-20 Quail", 
"A-10A Thunderbolt II",
"A-37A Dragonfly"
"WB-66D Destroyer", 
"B-1B Lancer", 
"B-29B Superfortress", 
"B-52D Stratofortress", 
"B-17G Flying Fortress", 
"C-141C Starlifter", 
"C-7A Caribou", 
"C-47B Skytrain", 
"C-54G Skymaster", 
"C-119C Flying Boxcar",
"C-124C Globemaster II",
"C-130E Hercules",
"EC-135N Stratotanker",
"VC-140B JetStar",
"UC-78B Bamboo Bomber",
"KC-97L Stratofreighter",
"C-46D Commando",
"C-123K Provider",
"EC-121K Constellation",
"F-80C Shooting Star",
"P-40N Warhawk",
"F-84E Thunderjet",
"F-89J Scorpion",
"F-101F Voodoo",
"F-105D Thunderchief",
"F-111E Aardvark",
"F-4D Phantom II",
"F-16A Fighting Falcon",
"F-15A Eagle",
"F-102A Delta Dagger",
"F-106A Delta Dart",
"F-86H Sabre",
"P-51H Mustang",
"F-100D Super Sabre",
"SR-71A Blackbird"]

plane_names = whitespaceToSpace(plane_raw)
print(plane_names)



for i in range(0, len(plane_names)):
    for j in range(0, 100, 20):
        planeFigs = photoCollection(plane_names[i], j)
        photoLoop(plane_raw[i], planeFigs)




    





