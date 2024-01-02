#%%
import requests
from time import time

import numpy as np

def norm(x,mean,std):
  return (x - mean) / std
# url = 'http://localhost:8501/v1/models/prediction_of_housing_quantity:predict'
url = 'https://37da3553.r1.cpolar.top/v1/models/prediction_of_housing_quantity:predict'


t1=norm(36,159,12.339623)
t2=norm(0,159,3.062893)
t3=norm(200,159,98.308176)
a = np.array([t1,t2,t3])



# train_stats
#          count       mean        std  min   25%   50%    75%    max
# 事业部数量    159.0  12.339623  11.244031  1.0   2.0  11.0   20.5   36.0
# weekday  159.0   3.062893   2.008481  0.0   1.0   3.0    5.0    6.0
# index    159.0  98.308176  57.795027  1.0  49.5  99.0  146.5  199.0



predict_request = '{"instances" : [%s]}' % list(a)

print(predict_request)
r = requests.post(url,data=predict_request)
print(r.content)