import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')


loansData['FICO.Score'] = loansData['FICO.Range'].map(str).map(lambda x: x.split('-')).map(lambda x: x[0]).map(int)

loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: x[:-1]).map(float)

loansData['Loan.Length'] = loansData['Loan.Length'].map(str).map(lambda x: x.replace(' months','')).map(int)

loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].map(str).map(lambda x: x.replace('%','')).map(float)

plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

loansData = loansData[['Amount.Requested','Amount.Funded.By.Investors','Interest.Rate','Debt.To.Income.Ratio','Inquiries.in.the.Last.6.Months', 'Employment.Length','FICO.Score']]

a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')



intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])

X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

f.summary()

## p value = 0.00 for both coefficients x1 and x2 as well as the constant. 
## R2 = .657 (decent not great)

