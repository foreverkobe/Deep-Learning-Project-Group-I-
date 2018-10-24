#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 22:44:30 2018

@author: yt
"""

import os
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import re
import urllib

import selenium



#import socket

#socket.setdefaulttimeout(3)

from selenium import webdriver

driver = webdriver.Chrome('/Users/yt/Desktop/Python/chromedriver')

'''
chromeOptions = webdriver.ChromeOptions()
prefs = {
    #"profile.default_content_settings.popups" : 0,
    "download.default_directory" : "/Users/yt/svgt/google_trends/"+word,
    "directory_upgrade": True,
    "extensions_to_open": ""
}
#prefs = {"download.default_directory" : "/Users/yt/svgt/google_trends/"+word}
chromeOptions.add_experimental_option("prefs",prefs)
driver = webdriver.Chrome(executable_path='/Users/yt/Desktop/Python/chromedriver', options=chromeOptions)
'''       



#word_list=['housing','credit','revenue','inflation','dow jones','return','markets','unemployment','growth','hedge']
#word_list=['revenue','inflation','dow jones','return','markets','unemployment','growth','hedge']

#word_list=['markets','unemployment','growth','hedge','money']
word_list=['revenue']

for word in word_list:
    '''
    for year in range(2007,2019):
        start_date=str(year)+'-04-22'
        end_date=str(year)+'-10-30'
        
        driver.get('https://trends.google.com/trends/explore?date='+start_date+'%20'+end_date+'&geo=US&q='+word);
        driver.refresh();
        time.sleep(5)
        
        driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/md-content/div/div/div[1]/trends-widget/ng-include/widget/div/div/div/widget-actions/div/button[1]/i').click()
        
        os.rename
    '''
    #for year in range(2006,2018):
    for year in range(2012,2018):
        start_date=str(year)+'-10-22'
        end_date=str(year+1)+'-04-30'
            
        driver.get('https://trends.google.com/trends/explore?date='+start_date+'%20'+end_date+'&geo=US&q='+word);
        driver.refresh();
        time.sleep(5)
        
        driver.find_element_by_xpath('/html/body/div[2]/div[2]/div/md-content/div/div/div[1]/trends-widget/ng-include/widget/div/div/div/widget-actions/div/button[1]/i').click()
    
    time.sleep(3)
    
    if not os.path.exists("/Users/yt/svgt/google_trends/"+word):
        os.mkdir("/Users/yt/svgt/google_trends/"+word)
    
    files = ['/Users/yt/Downloads/'+filename for filename in os.listdir('/Users/yt/Downloads') if filename.startswith("multiTimeline")]    
    #newname = ['/Users/yt/svgt/google_trends/'+word+'/'+filename for filename in os.listdir('/Users/yt/Downloads') if filename.startswith("multiTimeline")]
    
    for file in files:
        shutil.move(file,'/Users/yt/svgt/google_trends/'+word+'/')
    


driver.quit()
    
