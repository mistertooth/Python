Python Code
# BASIC CONCEPTS

# Series VS DataFrame
# DataFrame: Two-dimensional, heterogeneous tabular data structure with labeled axes (rows and columns). like a dict-like container for Series objects.
# Series: Series is the datastructure for a single column of a DataFrame

# Arrays VS Dataframe
# Arrays: can have any number of dimensions, but every entry has to have the same type.e.g. represent a 2 x 2 x 2 tensor, you need an array
# Data frames: are two-dimensional, but each column is allowed to have its own type.e.g. mix numerical and categorical data, you need a data-frame.


# import
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from numpy.random import randn

import matplotlib.pyplot as plt
import seaborn


# REINDEX
# create array
ser1 = np.arange(25)
#create a list within an int range
lst = range(1,6)
# create series with assigned index
ser1 = Series([1,2,3,4],index = ['A','B','C','D'])
# Reindex a series
ser2 = ser1.reindex(['A','B','C','D','E','F'])
# Reindex assigning unmatched index a value
ser2.reindex(['A','B','C','D','E','F','G'], fill_value=0)
# reindex with new list of index, fill extra rows with repetitive values as the previous row
ser3.reindex(lst3,method='ffill')
# select data by index from dataframe and assign with column names
ser4.ix[[2,3,4],['col2','col3']]
# select data by row and reassgin column names
ser4.ix[[2,3,4],col2
# index and columns numbers must align with the fields
df3 = DataFrame(ser1.reshape(5,5), index = lst, columns = col

# DROP VALUE BY INDEX OR COLUMN
# create series with auto increment integer till 3 with assigned index
ser1 = Series(np.arange(3), index = ['A','B','C'])
# create dataframe from series with assigned index (row name) and column
df1 = DataFrame(np.arange(9).reshape((3,3)), index = ['SF','LA','NY'], columns= ['pop','sizse','year'])
# drop row by index
df2 = df1.drop('LA')
# drop column, must specify axix num
df3 = df1.drop('sizse', axis=1)


# CSV
# define url
url_0511 = '/Users/shaokuixing/Desktop/whatever/survey50_300_left20160426.csv'
# read csv file without header
rawData = pd.read_csv(url_0511, sep='\t', thousands = '.', decimal =',', header=None)

# NULL NaN values
# check NaN and null values by column
rawData.isnull().any()
# check how many null values exist in devicemodelid
print ('no. rows with null deviceModelId :',rawData.devicemodelid.isnull().values.sum())
# show 5 random rows of data with devicemodel being null
rawData[rawData.devicemodelid.isnull()].sample(n=5, random_state=0)
# fill all null values with 0
rawData['viewtime50pct300ms']=rawData.viewtime50pct300ms.fillna(0)
rawData['devicemodelid']=rawData.devicemodelid.fillna(0
# check if there is null data at all
rawData.isnull().any().any()

# SQL sqlite3
import sqlite3
from pandas.io import sql
cnx = sqlite3.connect(':memory:', timeout=3000)
cnx.text_factory = str
# write to database
sql.to_sql(df5, name = 'df5', con=cnx, if_exists='replace')
# select data
sql.read_sql("select * from df5 limit 1",cnx)
# calculate time difference
SELECT
cast( (strftime('%s',t.finish)-strftime('%s',t.start)) AS real)/60/60 AS elapsed
FROM some_table AS t;


# SUBSET
# select subset based on column values condition OR
df3 = df3[(df3.format != 'takeover')|(df3.format != 'swipe')]
# select subset based on multiple column values AND
df.loc[(df["B"] > 50) & (df["C"] == 900), "A"]
df.loc[df['A'] > 2, 'B']

# DROP or DELETE
# DROP  unnecessary columns
recall.drop(recall[['ioptiontext','answerid']],axis=1, inplace=True)
# DROP rows based on condition
df = df[df.line_race != 0]
df = df[df.line_race.notnull()]
# DROP rows based on condition


# REPLACE
# LOC
df.loc[(df['A']!=2)& (df['B'].isnull()), 'C']=10
# np.where
df1 = df.where(((A>0) & (A<10)) | ((A>40) & (A<60))),1,0)
# replace
df['Á'] = df['Á'].replace(10, np.nan)
# record
record = {1:2, 2:3, 3:4}
df['A'] = df['A'].map(record)
# def function
# axis=1 means apply to every row
def function(x):
    if x['A'] == 1:
        return 2
    else
    return 0
df['A'] = df.apply(lambda x: function(x), axis=1)






# Ad labels of index
# ad columns
df.columns = ['a', 'b']


# print
# print static texts and dataframe
print("Total score for " + name + " is " + score)


# CALCULATE
# count values in a column or series
Series.value_counts(normalize=False, sort=True, ascending=False, bins=None, dropna=True)¶
pd.value_counts(df['categories'])



# NULL NaN
# fill null with 0 on underlying dataset
df[1].fillna(0, inplace=True)
# replace null value with mean of the values in the column
df.fillna(df.mean())
# check if any value is null in the dataframe
df.isnull().any().any()
# show which column contains null value_counts
df.isnull().any()
# count total number of null values
df.isnull().sum().sum()
# count null numbers by column
df.isnull().sum()

# DATATYPE
# category dataset
df["B"] = df["A"].astype('category')



# CUT
labels = [ "{0} - {1}".format(i, i + 9) for i in range(0, 100, 10) ]
df['group'] = pd.cut(df.value, range(0, 105, 10), right=False, labels=labels)
# BINNING
bins = [0, 25, 50, 75, 100]
group_names = ['Low', 'Okay', 'Good', 'Great']
categories = pd.cut(df['postTestScore'], bins, labels=group_names)
df['categories'] = pd.cut(df['postTestScore'], bins, labels=group_names)
# QCUT
df['A'] = pd.qcut(df.A, 4, lables=['25th','50th','75th','100th' ])
# CATEGORIZE
df = pd.Categorical(["a","b","c","a"], categories=["b","c","d"], ordered=False)
df1 = df.astype("category", categories=["b","c","d"], ordered=False)
# cat & set_categories
df["Status"] = df["Status"].astype("category")
df["Status"].cat.set_categories(["won","pending","presented","declined"],inplace=True)

# RESHAPE
# CROSSTAB
df2 = crosstab(df1.A, df1.B)
# PIVOT
# http://pbpython.com/pandas-pivot-table-explained.html
pd.pivot_table(df,index=["Manager","Rep"],values=["Price"],aggfunc=[np.mean,len])



# VISUALIZATION
# seaborn
# countplot
df4=df3[['market','uid']].reset_index()
seaborn.countplot(x='market', data=df4)
plt.xlabel('market')
plt.title('count by market')

# SELECT
# count unique group by column
df_test=df3.groupby('channel').answersetid.nunique().sort_values()
df_channel=pd.DataFrame(df_test).reset_index()
df_channel.columns=['channel','numAnswer']
