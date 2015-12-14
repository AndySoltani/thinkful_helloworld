import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf


df = pd.read_csv('/Users/andys/Dropbox (ThinkNear)/Andy S/Sublime Text/DataScience/thinkful/projects/hello_world/LoanStats3d_2015.csv', index_col=0)

#cleaning
df = df[1:(len(df)-2)]
df['int_rate'] = df['int_rate'].map(lambda x: x[:-1]).map(float)
df = df[df['home_ownership'] !='ANY']

#setting my explanatory and dependent variables
X = df['annual_inc']
y = df['int_rate']

## fit a OLS model with intercept on annual income
X = sm.add_constant(X)
est = sm.OLS(y, X).fit()

#est.rsquared_adj
#Out[25]: 0.0092888258027513659

df['log_annual_inc'] = np.log(df['annual_inc'])

Xlog = df['log_annual_inc']
Xlog = sm.add_constant(Xlog)
est_log = sm.OLS(y, Xlog).fit()

#In [37]: est_log.rsquared_adj
#Out[37]: 0.023785543437877754

###adding home ownership

df['home_ownership_ord'] = pd.Categorical(df.home_ownership,categories={'OWN':1,'MORTGAGE':2,'RENT':3}).labels
df['home_ownership_ord'].groupby(df['home_ownership_ord']).count()


#In [68]: df['home_ownership'].groupby(df['home_ownership']).count()
#home_ownership
#ANY              1
#MORTGAGE    142809
#OWN          31083
#RENT        116694

est_multi_1 = smf.ols(formula='int_rate ~ home_ownership_ord + log_annual_inc', data=df).fit()
#est_multi_1 = smf.ols(formula='int_rate ~ home_ownership_ord + annual_inc', data=df).fit()
est_multi = smf.ols(formula='int_rate ~ C(home_ownership) + log_annual_inc', data=df).fit()
est_multi_2 = smf.ols(formula='int_rate ~ C(home_ownership) + annual_inc', data=df).fit()



df['home_own_TF'] = df['home_ownership'].map(lambda x: 1 if x =='OWN' else 0)

est_TF =  smf.ols(formula='int_rate ~ home_own_TF + log_annual_inc', data=df).fit()

df['home_debt_TF'] = df['home_ownership'].map(lambda x: 1 if x !='OWN' else 0)

est_debt =  smf.ols(formula='int_rate ~ home_debt_TF + log_annual_inc', data=df).fit()

##none of these seem to help the adjusted R2

##interaction

est_interaction = smf.ols(formula='int_rate ~ C(home_ownership) * log_annual_inc', data=df).fit()

#In [151]: est_interaction.rsquared_adj
#Out[151]: 0.024514849209765321
#improvemnt to adjusted r-squared

est_interaction_1 = smf.ols(formula='int_rate ~ C(home_ownership) * annual_inc', data=df).fit()


