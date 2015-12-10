import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math


loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')


loansData['FICO.Score'] = loansData['FICO.Range'].map(str).map(lambda x: x.split('-')).map(lambda x: x[0]).map(int)

loansData['Interest.Rate'] = loansData['Interest.Rate'].map(lambda x: x[:-1]).map(float)

loansData['Loan.Length'] = loansData['Loan.Length'].map(str).map(lambda x: x.replace(' months','')).map(int)

loansData['Debt.To.Income.Ratio'] = loansData['Debt.To.Income.Ratio'].map(str).map(lambda x: x.replace('%','')).map(float)

loansData.to_csv('loansData_clean.csv', header=True, index=False)

df = pd.read_csv('loansData_clean.csv')

df['IR_TF'] = df['Interest.Rate'].map(lambda x: 1 if x < 12 else 0)

#check, should be 1s
#df[df['Interest.Rate'] == 13].head()

df['intercept'] = 1

ind_vars = ['Amount.Requested','FICO.Score','intercept']

logit = sm.Logit(df['IR_TF'], df[ind_vars])

result = logit.fit()

coeff = result.params
print(coeff)

#Amount.Requested     0.000174
#FICO.Score          -0.087423
#intercept           60.125045

#interest_rate = 60.125 - 0.087423(FicoScore) + 0.000174(LoanAmount)

#interest_rate = 60.125 - 0.087423*750.00 + 0.000174*10000.00

# p(x) = 1/(1 + e^-(60.125 - .087423 * FICO + .000174 * LoanAmount))

prob = 1/(1 + math.exp(-60.125 +.087423 * 750 - .000174 * 10000))

def logistic_function(FICO, Amount, coeff):
	p = 1/(1 + math.exp(coeff['intercept'] + coeff['FICO.Score'] * FICO + coeff['Amount.Requested'] * Amount))
	return p


#In [9]: logistic_function(720,10000,coeff)
#Out[9]: 0.25362141104848557

#logistic_function(750,10000,coeff)
#Out[10]: 0.024077937003110315


#p is below .7 for both; unlikely at both 750 and 720 credit scores that the loan will have interest rate < 12%

probs = []

for i,x in df.iterrows():
	probs.append(logistic_function(x['FICO.Score'],x['Amount.Requested'],coeff))


probs = np.asarray(probs)

df['probs'] = probs

df.plot('Interest.Rate','probs',kind='scatter')

df.plot('FICO.Score','probs',kind='scatter')

#how can I plot lines of fico scores on the below?
df.plot('Amount.Requested','probs',kind='scatter')

def pred(FICO, Amount, coeff):
	p = logistic_function(FICO, Amount, coeff)
	if p >= 0.70:
		return "yes you qualify"
	else:
		return "you do not qualify"

