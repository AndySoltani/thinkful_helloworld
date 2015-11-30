import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

loansData.dropna(inplace=True)

loansData.boxplot(column='Amount.Requested')
plt.savefig("plots/loansData_Requested_boxplot.png")

loansData.hist(column='Amount.Requested')
plt.savefig("plots/loansData_requested_hist.png")

plt.figure()
graph = stats.probplot(loansData['Amount.Requested'], dist="norm", plot=plt)
plt.savefig("plots/loansData_requested_QQplot.png")
plt.clf()

loansData.boxplot(column='Amount.Funded.By.Investors')
plt.savefig("plots/loansData__funded_boxplot.png")

loansData.hist(column='Amount.Funded.By.Investors')
plt.savefig("plots/loansData_funded_hist.png")

plt.figure()
graph = stats.probplot(loansData['Amount.Funded.By.Investors'], dist="norm", plot=plt)
plt.savefig("plots/loansData_funded_QQplot.png")