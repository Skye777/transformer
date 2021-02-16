# -*- coding: utf-8 -*-
import urllib
from bs4 import BeautifulSoup

rawurl = 'https://psl.noaa.gov/cgi-bin/db_search/DBListFiles.pl?did=98&tid=83478&vid=1917'
content = urllib.request.urlopen(rawurl).read().decode('ascii')  # 获取页面的HTML

soup = BeautifulSoup(content, 'html.parser')
url_cand_html = soup.find_all(class_='col-md-12')  # 定位到存放url的标号为content的div标签
list_urls = url_cand_html[1].find_all("a")  # 定位到a标签，其中存放着文件的url
urls = []

for i in list_urls:
    urls.append(i.get('href'))  # 取出链接

for i, url in enumerate(urls):
    print("This is file" + str(i + 1) + " downloading! You still have " + str(
        41 - i - 1) + " files waiting for downloading!!")
    file_name = "../meta-data/thflx/" + url.split('/')[-1]  # 文件保存位置+文件名
    urllib.request.urlretrieve(url, file_name)
