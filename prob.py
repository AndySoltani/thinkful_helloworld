import numpy as np 
import scipy.stats as stats
import matplotlib.pyplot as plt
import collections

testlist = [1, 4, 5, 6, 9, 9, 9]

c = collections.Counter(testlist)

# calculate the number of instances in the list
count_sum = sum(c.values())

for k,v in c.iteritems():
  print("The frequency of number " + str(k) + " is " + str(float(v) / count_sum))

testlist_plot, ax = plt.subplots()
ax.boxplot(testlist)
testlist_plot.savefig("plots/testlist_boxplot.png")

testlist_plot, ax = plt.subplots()
ax.hist(testlist)
testlist_plot.savefig("plots/testlist_hist.png")

testlist_plot, ax = plt.subplots()
graph1 = stats.probplot(testlist, dist="norm", plot=ax)
testlist_plot.savefig("plots/testlist_qqplot.png")


x_list = [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 4, 4, 4, 4, 5, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 9, 9]
x_list_plot, ax = plt.subplots()
ax.boxplot(x_list)
x_list_plot.savefig("plots/x_list_boxplot.png")

x_list_plot, ax = plt.subplots()
ax.hist(x_list)
x_list_plot.savefig("plots/x_list_hist.png")

x_list_plot, ax = plt.subplots()
graph1 = stats.probplot(x_list, dist="norm", plot=ax)
x_list_plot.savefig("plots/x_list_qqplot.png")

x = collections.Counter(x_list)
count_x = sum(x.values())
for k,v in x.iteritems():
  print("The frequency of number " + str(k) + " is " + str(float(v) / count_x))


test_data = np.random.normal(size=1000) 
test_data_plot, ax = plt.subplots()
ax.boxplot(test_data)
test_data_plot.savefig("plots/test_data_boxplot.png")

test_data_plot, ax = plt.subplots()
ax.hist(test_data)
test_data_plot.savefig("plots/test_data_hist.png")

test_data_plot, ax = plt.subplots()
graph1 = stats.probplot(test_data, dist="norm", plot=ax)
test_data_plot.savefig("plots/test_data_qqplot.png")

t = collections.Counter(test_data)
count_test_data = sum(t.values())
for k,v in t.iteritems():
  print("The frequency of number " + str(k) + " is " + str(float(v) / count_test_data))


test_data2 = np.random.uniform(size=1000) 
test_data2_plot, ax = plt.subplots()
ax.boxplot(test_data2)
test_data2_plot.savefig("plots/test_data2_boxplot.png")

test_data2_plot, ax = plt.subplots()
ax.hist(test_data2)
test_data2_plot.savefig("plots/test_data2_hist.png")

test_data2_plot, ax = plt.subplots()
graph1 = stats.probplot(test_data2, dist="norm", plot=ax)
test_data2_plot.savefig("plots/test_data2_qqplot.png")

t = collections.Counter(test_data2)
count_test_data2 = sum(t.values())
for k,v in t.iteritems():
  print("The frequency of number " + str(k) + " is " + str(float(v) / count_test_data2))
