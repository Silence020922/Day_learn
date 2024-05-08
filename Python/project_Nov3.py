import jieba
from matplotlib import pyplot as plt
from wordcloud import WordCloud
from PIL import Image
import numpy as np

def tcg(texts):
    cut = jieba.cut(texts)  #分词
    string = ' '.join(cut)
    return string

jieba.load_userdict(r"Download/userdict.txt")
txt = open("Download/二十大报告全文.txt",'r',encoding='utf-8').read()
st = open('Download/stopwords.txt','r',encoding='utf-8')
string=tcg(txt)
img = Image.open('Download/project1.png') #打开图片
img_array = np.array(img) #将图片装换为数组
stop_words = [line.strip('\n') for line in st.readlines()]
words = jieba.lcut(txt)
counts = {}
for word in words:
    if len(word) == 1: #排除单个字符（汉字）的分词结果
        if word not in stop_words:
            stop_words.append(word)
        continue
    else:
        if word not in stop_words:
            counts[word] = counts.get(word,0) + 1

#排序
items = list(counts.items())
items.sort(key=lambda x:x[1], reverse=True)      #排序
# 输出前20的高频词
for i in range(20):                              #输出前20个词
    word, count = items[i]
    print ("{0:<10}{1:>5}".format(word, count))

wc = WordCloud(
    background_color='white',
    width=800,
    height=600,
    mask=img_array, #设置背景图片
    font_path=' ',
    stopwords=stop_words
)
wc.generate_from_text(string)#绘制图片
plt.imshow(wc)
plt.axis('off')
plt.show()  #显示图片