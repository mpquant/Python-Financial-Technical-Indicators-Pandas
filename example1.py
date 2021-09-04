
from  hb_hq_api import *
from  MyTT import *


df=get_price('btc.usdt',count=120,frequency='1d');      #  1d:1day  4h:4hour   60m: 60min    15m:15min
CLOSE=df.close.values;  OPEN=df.open.values;   HIGH=df.high.values;   LOW=df.low.values   

MA5=MA(CLOSE,5)
MA10=MA(CLOSE,10)
CROSS_TODAY=RET(CROSS(MA5,MA10))

print(f'BTC5 MA5 { MA5[-1]}    BTC10日均线 {MA10[-1]}' )
print('MA5 CROSS MA10',CROSS_TODAY)
print('LAST 5 DAY EVERY CLOSE PRICE> MA10 ',EVERY(CLOSE>MA10,5) )

DIF,DEA,MACD=MACD(CLOSE)
print('MACD',DIF,DEA,MACD)
