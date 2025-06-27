# 导入所需的包
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import sys



import warnings
warnings.filterwarnings('ignore')


all_elect=pd.read_csv('../data/residential/all_elect.csv',index_col=0)
all_elect=all_elect.iloc[:,1:]
all_elect.index=pd.to_datetime(all_elect.index,unit='h',origin='2013-01-01 01:00:00')
all_weather=pd.read_csv('../data/residential/all_weather.csv',index_col=0)
all_weather.index=all_elect.index
all_weather=all_weather.iloc[:,1:]

all_elect=all_elect.resample("7d",origin='start').sum()
all_weather=all_weather.resample("7d",origin='start').mean()

all_elect=(all_elect-all_elect.min())/(all_elect.max()-all_elect.min())
all_weather=(all_weather-all_weather.min())/(all_weather.max()-all_weather.min())

for i in range(936):
    tmp_elect=all_elect
    tmp_weather=all_weather
    col=tmp_elect.columns[i]
    y = [tmp_elect[col],tmp_weather[col]]
    x = tmp_elect.index
    plt.figure()
    plt.plot(x,y[0],label=col+'elec'+str(i))
    plt.plot(x,y[1],label=col+'temp'+str(i))
    plt.legend()
    plt.savefig('../data/residential/parseDataImg/image{}.jpg'.format(i), dpi=600, bbox_inches='tight')
    print(i)
# 89 93
# 106
# 112
# 118
# 127
# 139
# 163
# 171
# 177
# 191
# 210
#
