import csv
import numpy as np
import re
import matplotlib.pyplot as plt
from matplotlib import font_manager
pic, axes = plt.subplots(nrows=2, ncols=2)  # 初始化画布
axes[0, 0].set(title='济南市各区总单数',  ylabel='成交数目')
axes[0, 1].set(title='济南市二手房成交周期')
axes[1, 0].set(title='济南市二手房年成交总量',ylabel = '年份')
axes[1, 1].set(title='济南市二手房年交易均价',ylabel = '年份')

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']


def readCSV(path):
    with open(path, "r", encoding='UTF-8') as file:
        data = csv.reader(file)
        list = []
        for row in data:
            list.append(row)
    return list


data_list = readCSV("/home/surplus/Downloads/secpond_price.csv")


# 地区交易总额
region_list = []  # 录入地区名(无重)
for i in np.arange(int(len(data_list)/10)):
    if data_list[i*10 + 1][1] not in region_list:
        region_list.append(data_list[i*10 + 1][1])


region_deal = [i[1] for i in data_list[1:]] # 录入地区名(有重)

total_deal = []# 计算地区交易总量
for i in region_list:
    total_deal.append(region_deal.count(i))

for i in range(len(region_list)): #绘图
    p1 = axes[0, 0].bar(region_list[i], total_deal[i], label='value')
    axes[0, 0].bar_label(p1, label_type='edge')

# 观察总体成交周期
deal_cycle_str = [re.sub('[\u4e00-\u9fa5]', '', i[17]) for i in data_list[1:]]
deal_cycle_int = [int(i) for i in deal_cycle_str] # 转化为整型
cycle_class = ['一个月以内','一个月至半年','半年以上','一年以上']
deal_cycle = [len([i for i in deal_cycle_int if i <=30]),
    len([i for i in deal_cycle_int if i > 30 and i<=180 ]),
    len([i for i in deal_cycle_int if i >180 and i<= 365]),
    len([i for i in deal_cycle_int if i >365])]
axes[0,1].pie(deal_cycle,
        labels=cycle_class, # 设置饼图标签
        colors=["#d5695d", "#5d8ca8", "#65a479", "#a564c9"], # 设置饼图颜色
        autopct='%.2f%%', # 格式化输出百分比,
        explode=(0, 0, 0, 0.2)
       )

#年份均价走势图
money_str = [re.sub('[\u4e00-\u9fa5,/]', '', i[10]) for i in data_list[1:]]
money_int = [int(i) for i in money_str]
deal_time_str = [re.sub('[\u4e00-\u9fa5]', '', i[8][0:4]) for i in data_list[1:]]
deal_time_int = [int(i) for i in deal_time_str]

zipped = zip(deal_time_int,money_int)  # 执行同步排序
sort_zip = sorted(zipped,key = lambda x:(x[0],x[1]))
uzip = zip(*sort_zip)
time_year, money = [list(x) for x in uzip]
year_total_num = []
for i in np.arange(time_year[0],time_year[-1] + 1):
    year_total_num.append(time_year.count(i))
 
axes[1,0].plot(np.arange(time_year[0],time_year[-1] + 1), year_total_num,label = '年成交总量 ',c =  'g') #年成交总量绘图
axes[1,0].legend()
year_mean_money = []

temp = 0
for j in year_total_num:
    year_mean_money.append(sum(money[temp:temp + j])/j)
    temp += j

axes[1,1].plot(np.arange(time_year[0],time_year[-1] + 1),year_mean_money,label = '年平均单价 元\平米',c = 'r') #年成交均价绘图
axes[1,1].legend()

plt.show()

# 寻找系统中可用字体
# import matplotlib
# print(matplotlib.matplotlib_fname())
# from matplotlib.font_manager import FontManager
# import subprocess

# fm = FontManager()
# mat_fonts = set(f.name for f in fm.ttflist)
# print (mat_fonts)
# output = subprocess.check_output('fc-list :lang=zh -f "%{family}\n"', shell=True)
# print ('*' * 10, '系统可用的中文字体', '*' * 10)
# print (output)
# zh_fonts = set(f.split(',', 1)[0] for f in output.decode().split('\n'))
# available = mat_fonts & zh_fonts
# print ('*' * 10, '可用的字体', '*' * 10)
# for f in available:
#     print(f)
