import pyodbc
import requests
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
import re

cnxn = pyodbc.connect('DRIVER={MySQL ODBC 3.51 Driver};SERVER=localhost;DATABASE=cheongwadae;UID=root;PWD=0000;charset=UTF8')
cursor = cnxn.cursor()

driver = webdriver.Chrome('C:\chromedriver_win32\chromedriver.exe')

link = 'https://www1.president.go.kr/petitions'
req = requests.get(link)
driver.get(link)

html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

webdriver.ActionChains(driver).send_keys(Keys.ESCAPE).perform() # close pop-up
# time.sleep(0.3)

petition_list = soup.select('div.bl_subject > a')

petition_num_list = []
for petition in petition_list:
    href_num = petition.get('href')[-6:]
    petition_num_list.append(href_num)

max_petition_num = max(petition_num_list)
temp = max_petition_num

while True:
    req = requests.get(link+'/'+temp)
    driver.get(link+'/'+temp)

    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')

    petition_title = soup.select('h3.petitionsView_title')
    petition_title = petition_title[0].get_text()
    petition_title = petition_title.replace("'", "")
    petition_body = soup.select('div.petitionsView_write')
    petition_body = petition_body[0].get_text()
    petition_body = petition_body.replace("'", "")
    petition_info_list = soup.select('div.petitionsView_info > ul')
    petition_info_list = petition_info_list[0].get_text().split('\n')
    petition_category = petition_info_list[1][4:]
    petition_start_date = petition_info_list[2][4:]
    petition_end_date = petition_info_list[3][4:]

    if petition_start_date != '2019-11-24':
        break

    cnxn.setdecoding(pyodbc.SQL_WCHAR, encoding='utf8')
    cnxn.setencoding(encoding='utf8')
    petition_body = petition_body[20:]

    cursor.execute("INSERT INTO petition VALUES (NULL, '%s', '%s', '%s', '%s', '%s')" % (petition_title, petition_body, petition_category, petition_start_date, petition_end_date))

    temp = str(int(temp) - 1)

cnxn.commit()

