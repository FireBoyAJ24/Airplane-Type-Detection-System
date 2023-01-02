from bing_image_downloader import downloader

import os

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
"AC-130A Spectre", 
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
"HH-43F Huskie",
"CH-21B Workhorse",
"UH-1P Iroquois",
"MH-53M Pave Low" ,
"SR-71A Blackbird"]

for plane_name in plane_raw:
    search = "https://www.google.com/search?q=" + plane_name + "&tbm=isch"


    dir_end = "dataset\\Aircraft Pictures\\" + plane_name 

    if not os.path.exists(dir_end):
        os.makedirs(dir_end)

    downloader.download(plane_name, limit=100,  output_dir=dir_end, adult_filter_off=True, force_replace=False, timeout=60, verbose=True)