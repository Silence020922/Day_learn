from urllib.request import urlopen
from urllib.parse import urlparse
from xml.etree.ElementInclude import include
from bs4 import BeautifulSoup 
from urllib.error import URLError
from urllib.error import HTTPError
import re
import time # 由于3.9以上版本random.seed支持格式改变，暂用time库平替。

def bs4_pre(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    except URLError as e :
        return  None
    try:
        bs = BeautifulSoup(html.read(),'html.parser')
    except AttributeError as e:
        return None
    return bs

def getmathweb(startingPage):
    bs = bs4_pre(startingPage)
    mainsite = '{}://{}'.format(urlparse(startingPage).scheme,urlparse(startingPage).netloc)
    allthing = bs.find_all('ul',{'class':'list'})
    linklist = []
    for thing in allthing:
            alllink = re.findall(r"href=\"(.*?)\"",thing.prettify())
            thinglist = thing.get_text().split('\n')	
    thinglist.pop(0)
    for link in alllink:
        if re.match('info',link) == None:
            link = link.replace('&amp;','&')
            linklist.append(link)
        else:
            link = mainsite + '/' + link
            linklist.append(link)
    dictionary = dict(zip(thinglist, linklist))
    for i,j in dictionary.items():
        print(i,j)
