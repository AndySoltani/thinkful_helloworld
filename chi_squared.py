from scipy import stats
import collections
import pandas as pd
import matplotlib.pyplot as plt


loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData.dropna(inplace=True)

freq = collections.Counter(loansData['Open.CREDIT.Lines'])

len(freq.keys())

plt.figure()
plt.bar(freq.keys(), freq.values(), width=1)
plt.show()

chi, p = stats.chisquare(freq.values())

print "the chi square value of the chi-squared test is {0}, corresponding to a p value of {1}".format(chi,p)

freq2 = collections.Counter(loansData['Employment.Length'])

len(freq2.keys())

#plt.figure()
#loansData.hist(column='Employment.Length')
#plt.show()

chi, p = stats.chisquare(freq2.values())

print "the chi square value of the chi-squared test is {0}, corresponding to a p value of {1}".format(chi,p)