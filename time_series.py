#2007-2011 data

import pandas as pd
import numpy as np 

df = pd.read_csv('/Users/andys/Dropbox (ThinkNear)/Andy S/Sublime Text/DataScience/thinkful/LoansData_2007to2011_clean.csv', header=0, low_memory=False)


#cleaning
df = df[1:(len(df)-2)]
df['int_rate'] = df['int_rate'].map(lambda x: x[:-1]).map(float)
df = df[df['home_ownership'] !='ANY']

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d'],format='%b-%y')
dfts = df.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

import matplotlib.pyplot as plt
loan_count_summary.plot()


#2015 data
import pandas as pd
import numpy as np 

df = pd.read_csv('/Users/andys/Dropbox (ThinkNear)/Andy S/Sublime Text/DataScience/thinkful/LoanStats3d_2015.csv', header=0, low_memory=False)


#cleaning
df = df[1:(len(df)-2)]
df['int_rate'] = df['int_rate'].map(lambda x: x[:-1]).map(float)
df = df[df['home_ownership'] !='ANY']
df=df[pd.notnull(df['issue_d'])]

# converts string to datetime object in pandas:
df['issue_d_format'] = pd.to_datetime(df['issue_d'],format='%b-%Y')
dfts = df.set_index('issue_d_format') 
year_month_summary = dfts.groupby(lambda x : x.year * 100 + x.month).count()
loan_count_summary = year_month_summary['issue_d']

import matplotlib.pyplot as plt
loan_count_summary.plot()

##non-stationary model as it increases every year and even monthly

##need to difference out the yearly growth


import statsmodels.api as sm

sm.graphics.tsa.plot_acf(loan_count_summary)
sm.graphics.tsa.plot_pacf(loan_count_summary)

#auto correlation is growing over time, not decaying, indicating a non-stationary series that has to be differenced.
#partial auto correlation is also growing and not decaying over time. 

