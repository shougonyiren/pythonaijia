#%%
import requests
from time import time

import numpy as np

def norm(x,mean,std):
  return (x - mean) / std
url = 'http://localhost:8501/v1/models/prediction_of_housing_quantity:predict'


t1=norm(36,154,11.344156)
t2=norm(0,154,3.051948)
t3=norm(192,154,93.935065)
a = np.array([t1,t2,t3])

predict_request = '{"instances" : [%s]}' % list(a)

r = requests.post(url,data=predict_request)
print(r.content)


#  count    mean    std         min   25%   50%     75%    max
# 事业部数量 154.0  11.344156  10.540541  1.0   2.0  11.0   15.75   36.0
# weekday  154.0   3.051948   1.986201  0.0   1.0   3.0    5.00    6.0
# index    154.0  93.935065  55.822978  0.0  46.5  95.5  138.75  191.0