from scipy import stats
import pandas as pd

data = '''Region,Alcohol,Tobacco
North,6.47,4.03
Yorkshire,6.13,3.76
Northeast,6.19,3.77
East Midlands,4.89,3.34
West Midlands,5.63,3.47
East Anglia,4.52,2.92
Southeast,5.89,3.20
Southwest,4.79,2.71
Wales,5.27,3.53
Scotland,6.08,4.51
Northern Ireland,4.02,4.56'''

data = data.splitlines()

#data.split('\n')

data = [i.split(',') for i in data]

column_names = data[0]
data_rows = data[1::]
df = pd.DataFrame(data_rows, columns=column_names)

df['Alcohol'] = df['Alcohol'].astype(float)
df['Tobacco'] = df['Tobacco'].astype(float)

print "The mean of the Alcohol dataset is " + str(df['Alcohol'].mean())
print "The median of the Alcohol dataset is " + str(df['Alcohol'].median())
print "The mode of the Alcohol dataset is " + str(stats.mode(df['Alcohol'])[0][0])
print "The range of the Alcohol dataset is " + str(max(df['Alcohol']) - min(df['Alcohol']))
print "The standard deviation of the Alcohol dataset is " + str(df['Alcohol'].std())
print "The variance of the Alcohol dataset is " + str(df['Alcohol'].var())

print "The mean of the Tobacco dataset is " + str(df['Tobacco'].mean())
print "The median of the Tobacco dataset is " + str(df['Tobacco'].median())
print "The mode of the Tobacco dataset is " + str(stats.mode(df['Tobacco'])[0][0])
print "The range of the Tobacco dataset is " + str(max(df['Tobacco']) - min(df['Tobacco']))
print "The standard deviation of the Tobacco dataset is " + str(df['Tobacco'].std())
print "The variance of the Tobacco dataset is " + str(df['Tobacco'].var())
