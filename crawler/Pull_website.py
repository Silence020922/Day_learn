from DrissionPage import ChromiumPage
import csv
import time
"""
目标：以济南大明湖景区为出发点，辐射状获取济南本地的n个酒店(由于计算机处理能力有限，暂时不考虑获取济南全地区酒店信息)
"""

dp = ChromiumPage()
f = open('xc_data.csv',mode = 'w', encoding='utf-8', newline='')

csv_write = csv.DictWriter(f, fieldnames=[
            '酒店名称',
        '评价数量',
        '价格',
        '门牌号',
        '地区',
        '维度',
        '经度',
        '总体评价',
        '总体评分',
        '环境',
        '卫生',
        '服务',
        '设施',
        '标签'
])
csv_write.writeheader()
# 监听
dp.listen.start('json/HotelSearch')

# 访问网站
dp.get('https://hotels.ctrip.com/hotels/list?countryId=1&city=144&provinceId=0&checkin=2024/08/30&checkout=2024/08/31&lat=36.6752243&lon=117.0249786&optionId=4195720&optionType=Markland&directSearch=0&optionName=%E5%A4%A7%E6%98%8E%E6%B9%96%E6%99%AF%E5%8C%BA&display=%E5%A4%A7%E6%98%8E%E6%B9%96%E6%99%AF%E5%8C%BA%2C%20%E6%B5%8E%E5%8D%97%2C%20%E5%B1%B1%E4%B8%9C%2C%20%E4%B8%AD%E5%9B%BD&crn=1&adult=1&children=0&markland=%E5%A4%A7%E6%98%8E%E6%B9%96%E6%99%AF%E5%8C%BA&searchBoxArg=t&travelPurpose=0&ctm_ref=ix_sb_dl&domestic=1&')

# 翻页
for i in range(100):
    if i > 4: # 触发验证机制
        next_page = dp.ele('css:.btn-box span')
        if next_page.text == '搜索更多酒店':
            next_page.click()
    resp = dp.listen.wait()
    # 获取响应数据
    json_data = resp.response.body

    # 数据解析
    hotellist = json_data['Response']['hotelList']['list']

    for msg in hotellist:
        dir = {
            '酒店名称':msg['base']['hotelName'],
            '评价数量':(msg['comment']).get('content',None),
            '价格':msg['money']['price'],
            '门牌号':msg['position']['address'],
            '地区':msg['position']['area'],
            '维度':msg['position']['lat'],
            '经度':msg['position']['lng'],
            '总体评价':msg['score']['desc'],
            '总体评分':msg['score']['number'],
            '环境':msg['score']['subScore'][0]['number'],
            '卫生':msg['score']['subScore'][1]['number'],
            '服务':msg['score']['subScore'][2]['number'],
            '设施':msg['score']['subScore'][3]['number'],
            '标签':' '.join(msg['base']['tags'])
        }
        csv_write.writerow(dir)
    # time.sleep(1) # 防止被墙
    dp.scroll.to_bottom() # 页面下滑


dp = ChromiumPage()
f = open('mt_data3.csv',mode = 'w', encoding='utf-8', newline='')
csv_write = csv.DictWriter(f, fieldnames=[
        '酒店名称',
        '评价数量',
        '价格',
        '门牌号',
        '地区',
        '维度',
        '经度',
        '总体评价',
        '总体评分',
        '历史消费',
        '酒店电话',
        '标签',
        '酒店类型',
        '预定情况'
])
csv_write.writeheader()
dp.listen.start('hbsearch/HotelSearch')
dp.get('https://i.meituan.com/awp/h5/hotel/list/list.html?cityId=96&checkIn=2024-08-30&checkOut=2024-08-31&lat=36.793072&lng=119.949358&keyword=%E4%BD%93%E8%82%B2%E4%B8%AD%E5%BF%83&accommodationType=1&sort=smart')
for i in range(100):
    print('now the idx is {} :\n'.format(i))
    resp = dp.listen.wait()
    json_data = resp.response.body
    hotellist = json_data['data']['searchresult']
    for msg in hotellist:
        dir = {
            '酒店名称':msg['name'],
            '评价数量':(msg.get('commentsCountDesc',None)),
            '价格':msg.get('lowestPrice',None),
            '门牌号':msg['addr'],
            '地区':msg['areaName'],
            '维度':msg['lat'],
            '经度':msg['lng'],
            '总体评价':msg.get('avgScoreDesc',None),
            '总体评分':msg.get('avgScore',None),
            '历史消费':msg['historySaleCount'],
            '酒店电话':msg['forward'].get('phoneList',{0:{'phone':None}})[0]['phone'],
            '标签':' '.join(msg['poiAttrTagList'] + msg['forward']['serviceTagList']),
            '酒店类型':msg.get('hotelStar',None),
            '预定情况':msg.get('informationA',{0:{'text':None}})[0]['text']
        }
        csv_write.writerow(dir)
    time.sleep(0.5)
    dp.scroll.to_bottom() # 页面下滑

dp = ChromiumPage()
f = open('qn_data.csv',mode = 'w', encoding='utf-8', newline='')
csv_write = csv.DictWriter(f, fieldnames=[
            '酒店名称',
            '评价数量',
            '价格',
            '位置信息',
            '地区',
            '维度',
            '经度',
            '总体评价',
            '总体评分',
            '开业年份',
            '酒店类型',
            '标签',
            '特殊说明'
])
csv_write.writeheader()
dp.get('https://touch.qunar.com/hotelcn/jinan/q-%E5%A4%A7%E6%98%8E%E6%B9%96%E6%99%AF%E5%8C%BA?city=%E6%B5%8E%E5%8D%97&cityUrl=jinan&keywords=%E5%A4%A7%E6%98%8E%E6%B9%96%E6%99%AF%E5%8C%BA&checkInDate=2024-08-30&checkOutDate=2024-08-31&sort=0')
dp.listen.start('hotelcn/api/hotellist')
dp.scroll.to_bottom()
for i in range(50):
    print('now the idx is {} :\n'.format(i))
    resp = dp.listen.wait()
    json_data = resp.response.body
    hotellist = json_data['data']['hotels']

    for msg in hotellist:
        dir = {
            '酒店名称':msg['name'],
            '评价数量':(msg.get('commentCount',None)),
            '价格':msg.get('price',None),
            '位置信息':msg['locationInfo'],
            '地区':msg['hotPoi'],
            '维度':msg['gpoint'][0],
            '经度':msg['gpoint'][1],
            '总体评价':msg.get('commentDesc',None),
            '总体评分':msg.get('score',None),
            '开业年份':msg.get('whenFitment',None),
            '酒店类型':msg.get('dangciText',None),
            '标签':None,
            '特殊说明':None
        }
        if msg['newMedalAttrs'] :
            dir['特殊说明'] = msg['newMedalAttrs'][0]['title']
        tag = []
        labels_dir = msg.get('labels',[])
        if len(labels_dir) > 0:
            for label in labels_dir[1:]:
                tag.append(label['label'])
            dir['标签'] = ' '.join(tag)
        csv_write.writerow(dir)
    time.sleep(0.5)
    dp.scroll.to_bottom() # 页面下滑