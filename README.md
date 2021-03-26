# Data-Science-Portfolio
## Project 1. How to Use Online ESG Scores ([Blog](https://yifang-lin.medium.com/how-to-use-online-esg-scores-6620c645213))
### Installations
```
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import requests
from bs4 import BeautifulSoup
import ipywidgets as widgets
from ipywidgets import interact
```
### Motication
I am interested in obtaining online ESG data in order to form a top-down overview of how different sectors and countries are performing. 
1. How are total ESG scores constructed?
2. Which sectors and industries are among the best and worst performers in terms of ESG risks?
3. Can controversies scores be predicted using a linear regressional machine learning model based on ESG scores and sectors?
4. Are there any geographical patterns of controversy scores among S&P 500 firms?
### Files Explained
There are 4 files in Project 1. 
1. Part-1-Gathering-ESG-data.ipynb is where the scraping of ESG data takes place. 
2. Part-1-Gathering-S&P500-data.ipynb scrapes firm information from Wikipedia. 
3. Part-2-to-6-Analytics.ipynb contains data cleaning, analysis, modeling, etc. 
4. mapping.ipynb is a file that cleans the data and gets it ready for Tableau analysis. 
### Summary of Analysis
1. The Total ESG Scores are not a simple sum of a firm's E/S/G ratings, sectors are also a factor. 
2. Environment has the highest weight among all. One potential reason behind this is that environmental factors are more measurable and has longer history in sustainable finance. While the quantitative measures in social and governance aspects are still catching up. 
3. Real estate, information technology and consumer discretionary are among the best (lowest ESG scores), while energy, utilities, and materials ranked highest in terms of overall ESG risks. Interestingly, although overall environmental scores are the dominant factor of high total ESG scores, there are some high overall risk are mostly a result of high social or governance scores, such as healthcare, financials and communication services.
4. S&P 500 firms does not provide a robust comparison of different countries, and the UK has the lowest controversy score among the 5 countries mentioned.
### Acknowledgement
Thanks to the inspirations from the below authors I was able to turn my initial curiosity into something more informative:
- [Scrapping Financial (ESG) Data with Python](https://curt-beck1254.medium.com/scrapping-financial-esg-data-with-python-99d171a12c51)
- [How to Get S&P 500 Companies from Wikipedia](https://codingandfun.com/python-scraping-how-to-get-sp-500-companies-from-wikipedia/)
