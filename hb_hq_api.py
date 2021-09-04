# HUOBI 
import json,requests,urllib3,datetime;   import pandas as pd;  import numpy as np ; urllib3.disable_warnings()

huobi_domain='api.huobi.pro'  

class Context:   
    def __init__(self):  
        self.current_dt=datetime.datetime.now()

#1d:1DAT  4h:4HOUR   60m: 60MIN    15m:15MIN
def get_price(code, end_date=None, count=1,frequency='1d', fields=['close']):    
    code=code.replace('.','')
    frequency=frequency.replace('d','day').replace('m','min').replace('h','hour')
    url = f'https://{huobi_domain}/market/history/kline?period={frequency}&size={count}&symbol={code}'
    res = requests.get(url,verify=False).text  
    df=pd.DataFrame(json.loads(res)['data']);      df=df.iloc[::-1]    
    df["time"] = pd.to_datetime(df["id"]+8*3600, unit='s')              
    df=df[['time','open','close','high','low','vol']]                   
    df.set_index(["time"], inplace=True);     df.index.name=''         
    return df

def get_last_price(code):       
    return get_price(code,count=1,frequency='1m', fields=['close']).close[0]

def attribute_history(security, count, unit='1d', fields=['open', 'close', 'high', 'low', 'volume', 'money']):   
    return get_price(security = security, end_date = context.current_dt, frequency=unit, fields=fields, fq = fq, count = count)[:-1]            


