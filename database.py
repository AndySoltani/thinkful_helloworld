from sys import argv

import pandas as pd
import sqlite3 as lite

month = argv[1]

con = lite.connect('getting_started.db')

cities = (
	('New York City', 'NY'),
    ('Boston', 'MA'),
    ('Chicago', 'IL'),
    ('Miami', 'FL'),
    ('Dallas', 'TX'),
    ('Seattle', 'WA'),
    ('Portland', 'OR'),
    ('San Francisco', 'CA'),
    ('Los Angeles', 'CA'),
    ('Las Vegas', 'NV'),
    ('Atlanta', 'GA')
    )

weather = (
('New York City',2013, 'July','January',62),
('Boston',2013,'July','January',59),
('Chicago',2013,'July','January',59),
('Miami',2013,'August','January',84),
('Dallas',2013,'July','January',77),
('Seattle',2013,'July','January',61),
('Portland',2013,'July','December',63),
('San Francisco',2013,'September','December',64),
('Los Angeles',2013,'September','December',75),
('Las Vegas', 2013, 'July', 'December',75),
('Atlanta', 2013, 'July', 'January',56)
)

query = "select city, state from cities inner join weather on name = city where warm_month = '{0}'".format(month)

with con:

    cur = con.cursor()
    cur.execute("Drop table if exists cities")
    cur.execute("Drop table if exists weather")
    cur.execute("create table cities (name text, state text)")
    cur.execute("create table weather (city text,year integer,warm_month text, cold_month text,average_high integer)")
    cur.executemany("INSERT INTO cities VALUES(?,?)", cities)
    cur.executemany("INSERT INTO weather VALUES(?,?,?,?,?)", weather)
    #cur.execute("select city, state, year, warm_month, cold_month, average_high from cities inner join weather on name = city")
    cur.execute(query)
    rows = cur.fetchall()
    cols = [desc[0] for desc in cur.description]
    df = pd.DataFrame(rows, columns=cols)


city_list = []
for i,j in df.iterrows():
	city_list.append((str(j['city']),str(j['state'])))

city_list_string = ""
for city in city_list:
	city_list_string += (city[0] + " " + city[1] + ", ")

city_list_string = city_list_string.rstrip()
city_list_string = city_list_string.rstrip(",")

print "The cities that are warmest in {1} are {0}".format(city_list_string,month)
