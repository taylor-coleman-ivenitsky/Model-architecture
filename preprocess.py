import time
import numpy as np
from PIL import Image
import pandas as pd
from selenium import webdriver

def preprocess(i):
    URL_START = 'https://www.archinform.net/projekte/'
    URL_END = '.htm'

    #browser = webdriver.Chrome(executable_path='/Users/taylorcoleman/PycharmProjects/scraper/venv/lib/python3.7/site-packages/selenium/webdriver/chrome/chromedriver')
    browser = webdriver.Chrome()

    NTSC_formula = [0.299, 0.587, 0.114] #used for grayscale conversion
    dates = []
    images = []

    for i in range(i, i+120): 
        CURR_URL = URL_START + str(i) + URL_END

        browser.get(CURR_URL)
        time.sleep(1)

        image = browser.find_elements_by_xpath('//div[@class="slick-slide slick-current slick-active slick-center"]')
        image_path = 'Images/image' + str(i) + '.png'

        if len(image) < 1:
            continue #only process the building if image data is available

        #get html for the table that stores date information for the building
        date_table = browser.find_elements_by_id('MenueCHROChild')

        #to get date, split on whitespace, take first four characters of third word, or fourth if "ca." is present
        if len(date_table) <= 0:
            continue #skip processing on any images without date

        if date_table[0].text.split()[2][0:3] == "ca.": #account for date format "ca. 1998"
            date = date_table[0].text.split()[3][0:4]
        else:
            date = date_table[0].text.split()[2][0:4]

        if not date.isnumeric():
            continue #skip processing on any images with incorrect date format

        dates.append(date)

        #saves image to the Images directory
        image[0].screenshot(image_path)

        #convert image to array
        image = Image.open(image_path)
        image = np.asarray(image)

        #convert to grayscale using standard NTSC formula
        gray_image = image[:,:,0] * NTSC_formula[0] + image[:,:,1] * NTSC_formula[1] + image[:,:,2] * NTSC_formula[2]

        #TODO: crop image by removing some of the top and bottom rows

        #rewrite grayscale, cropped image to Images folder to manually verify grayscale conversion and cropping
        gray_image_check = Image.fromarray(gray_image)
        gray_image_check = gray_image_check.convert("L") #reduces image from 3 to 1 channel (R, G, B) -> (B/W)
        gray_image_path = 'Images/image' + str(i) + 'grey.jpeg'
        gray_image_check.save(gray_image_path)
        datefile.write(str(i) + ", " + '\n')

        images.append(gray_image)

    # datefile = open("/dates.txt","a")
    # datefile.write(i, i+120, str(dates))
    # datefile.close()
    # print("dates", dates)
    return (images, dates)

datefile = open("dates.txt", "w")
for i in range(1, 40000, 120):
    preprocess(i)
datefile.close()
