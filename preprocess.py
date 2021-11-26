from bs4 import BeautifulSoup
import requests
import time
import numpy as np
from PIL import Image
import pandas as pd

URL_START = 'https://www.archinform.net/projekte/'
URL_END = '.htm'

NTSC_formula = [0.299, 0.587, 0.114] #used for grayscale conversion
results = []

for i in range(1, 2): #TODO: change upperbound to 89143 after debugging to get all images
    CURR_URL = URL_START + str(i) + URL_END

    r = requests.get(CURR_URL)
    r.raise_for_status()

    soup = BeautifulSoup(r.text, 'html.parser')
    imageInfo = soup.find('body')

    #TODO: populate these after getting HTML that makes sense
    date = ?
    image = ? #stored on website as jpg
    caption = ?

    #preprocess date text to standardized format
    date = int(date) #converts year from string to integer

    #reshaping - assuming for now that they are the same size but can add reshaping in pretty easily later if necessary
    image = Image.open(image)
    image = np.asarray(image)

    #convert to grayscale using standard NTSC formula
    gray_image = [:,:,0] * NTSC_formula[0] + [:,:,1] * NTSC_formula[1] + [:,:,2] * NTSC_formula[2]

    results.append((gray_image, date, caption))

#store data in file so that preprocessing only needs to be run once
pd.DataFrame(results).to_csv('data.csv')
