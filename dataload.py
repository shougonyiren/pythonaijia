import pandas as pd
import seaborn as sns



# def week_to_chinese(week_day):
#     chinese_week = {
#         0: '星期一',
#         1: '星期二',
#         2: '星期三',
#         3: '星期四',
#         4: '星期五',
#         5: '星期六',
#         6: '星期日'
#     }
#     return chinese_week[week_day]

# data_j= pd.read_excel(r'Z:\爱家工作\项目py数据\VR自研数据.xlsx',sheet_name='数据集')
data_j= pd.read_excel(r'/mnt/z/爱家工作/项目py数据/VR自研数据.xlsx',sheet_name='数据集')

# data_j.groupby(['订单创建日期'])[['房源id']].count()
# print(data_j.groupby(['订单创建日期'])[['房源id']].count().head(10))
print(data_j.groupby(['订单创建日期'])[['房源id']].count())
# data_j.groupby('订单创建日期').head(10).to_excel(r'/Volumes/ZHITAI_2TB/aijia工作/项目数据/test1.xlsx',index=False)






count=data_j.groupby(['订单创建日期'])[['房源id']].count().copy()#数量
count["拍摄数量"] =count['房源id']

print("count")
# print(count)

count["事业部数量"]=data_j.groupby(['订单创建日期'])[['所属事业部']].nunique().copy()#事业部数量
# print(data_j.groupby(['订单创建日期'])[['所属事业部']].nunique())#事业部数量




count.reset_index(inplace=True)

# print(count[1:3])
# print(count.iloc[2:]) #取第几条数据


# count['date'] = pd.to_datetime(count['订单创建日期'])
count['weekday'] = pd.to_datetime(count['订单创建日期']).dt.weekday

# count['weekday_name'] = count['date'].dt.day_name()

# print(count)

count['星期'] = count['weekday'].replace({0: '星期一',1: '星期二',2: '星期三',3: '星期四', 4: '星期五',5: '星期六',6: '星期日'})
# print(count)
count['index']=count.index
count = count.iloc[1:]#移除第一行 第一条是测试数据
count = count.iloc[:-1]#移除最后一行 最后一条数据不全
print(count)

count.to_excel(r'/mnt/z/爱家工作/项目py数据/count.xlsx',index=False)
# count.to_excel(r'Z:\爱家工作\项目py数据\count.xlsx',index=False)
# dataset = data_j.copy()

# print(dataset.tail())
# print(dataset.isna().sum())

