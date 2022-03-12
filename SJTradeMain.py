'''

저가 매수, 고가 매도 트레이딩.

연속 2일 간의 수정 종가 차이를 계산한다.
이 차이 값이 음수이면 전날 가격이 다음날 가격보다 높았기 대문에 가격이 낮아지므로 매수.
이 값이 양수이면 가격이 높기 때문에 매도.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data

start_date = '2014-01-01'
end_date = '2018-01-01'

# Date 항목이 여기서 인덱스다. 
goog_data = data.DataReader('GOOG', 'yahoo', start_date, end_date) 

# Date 항목이 인덱스이기 때문에 column을 넘겨주면 안되고, 명시적으로 .index로 Date 값을 전달해야 한다.
goog_data_signal = pd.DataFrame(index=goog_data.index)

# pandas DataFrame을 생성 후에, column 값을 추가한다.
goog_data_signal['price'] = goog_data['Adj Close']

# pandas의 diff() 함수를 이용하여 행과 행의 차이 값을 계산한다. (1행 - 0행, 2행 - 1행 ...)
goog_data_signal['daily_difference'] = goog_data_signal['price'].diff()

# signal column의 default 값을 0.0으로 설정.
goog_data_signal['signal'] = 0.0

# signal column의 값을 numpy 의 where() 함수를 이용하여 0보다 클 경우 1.0, 작을 경우 0.0 으로 설정.
# 아래와 같이 사용하기도 함.
# goog_data_signal['signal'][:] = np.where(goog_data_signal['daily_difference'][:] > 0, 1.0, 0.0)
goog_data_signal['signal'] = np.where(goog_data_signal['daily_difference'] > 0, 1.0, 0.0)

# positions column을 채운다. 매수, 매도 시그널의 값의 차이를 확인하여
# 하락장에서 계속 매수하거나, 상승장에서 계속 매도하는 것을 막는다.
# 1은 주식을 매수한 시점.
# 0은 아무것도 하지 않은 상태
# -1은 주식을 매도한 시점.
goog_data_signal['positions'] = goog_data_signal['signal'].diff()

# 그래프를 출력할 figure 객체를 만든다.
fig = plt.figure()

# add_subplot 함수의 첫번째 인자는 그리드의 어디 위치에 서브 플롯을 추가할 것인가를 의미한다.
# https://www.delftstack.com/ko/howto/matplotlib/add-subplot-to-a-figure-matplotlib/ 참고
# y축에 Google price in $ 라는 문자를 출력한다.
ax1 = fig.add_subplot(111, ylabel = 'Google price in $')

# plot 함수의 x축은 DataFrame의 index 값이다. 
# plot 함수의 첫번째 매개변수는 figure에 그려질 플롯의 모양을 의미한다. 
# color는 그래프의 색상을 의미한다.
# lw = line width
goog_data_signal['price'].plot(ax = ax1, color = 'r', lw = 2.)

# 매수한 위치에 위쪽 화살표를 그린다.
# 현재 price에 대한 그래프이므로 y는 price / x는 date 다.
# goog_data_signal DataFrame에서 positions 항목이 1인 경우의 index 위치를 x좌표
# goog_data_signal DataFrame에서 positions 항목이 1인 경우의 price 위치를 y좌표
ax1.plot(goog_data_signal.loc[goog_data_signal.positions == 1.0].index,
         goog_data_signal.price[goog_data_signal.positions == 1.0],
         '^', markersize=5, color='m')

ax1.plot(goog_data_signal.loc[goog_data_signal.positions == -1.0].index,
         goog_data_signal.price[goog_data_signal.positions == -1.0],
         'v', markersize=5, color='k')

#plt.show()

# 초기 투자자본
initial_capital = float(1000.0)

# signal 에 대한 데이터 프래임 생성. (원래는 매수, 매도 시점에 대해서만 해야하지 않나?)
position = pd.DataFrame(index = goog_data_signal.index).fillna(0.0)
portfolio = pd.DataFrame(index = goog_data_signal.index).fillna(0.0)

# # signal 값에 대한 것이 아니라 positions 값에 대해서 해야할 것 같은데...
# positions['GOOG'] = goog_data_signal['signal']

# # 가격과 실제 매수
# portfolio['positions'] = (positions.multiply(goog_data_signal['price'], axis = 0))
# portfolio['cash'] = initial_capital - (positions.diff().multiply(goog_data_signal['price'], axis=0)).cumsum()
# portfolio['total'] = portfolio['positions'] + portfolio['cash']
# portfolio.plot()
# plt.show()

# signal 값에 대한 것이 아니라 positions 값에 대해서 해야할 것 같은데...
position['GOOG'] = goog_data_signal['positions']

portfolio['positions'] = (position.multiply(goog_data_signal['price'], axis = 0))
portfolio['cash'] = initial_capital - (position.diff().multiply(goog_data_signal['price'], axis=0)).cumsum()
portfolio['total'] = portfolio['positions'] + portfolio['cash']
portfolio.plot()
plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(111, ylabel='Portfolio value in $')
# portfolio['total'].plot(ax=ax1, lw=2.)
# ax1.plot(portfolio.loc[goog_data_signal.positions == 1.0].index,portfolio.total[goog_data_signal.positions == 1.0],'^', markersize=10, color='m')
# ax1.plot(portfolio.loc[goog_data_signal.positions == -1.0].index,portfolio.total[goog_data_signal.positions == -1.0],'v', markersize=10, color='k')
# plt.show()