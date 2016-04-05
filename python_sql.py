# python_sql

# import lib
import pandas as pd
import numpy as np
from pandas import *
import psycopg2 as pg
import pandas.io.sql as psql

# connect to redshift
conn = pg.connect(database="logs", user="XXX", password="XXX",
                             host="XXX", port="5439")
print "connection successful"

# run query
surveyQ = psql.read_sql("select * from public.surveydb_SurveyQuestions_Languages limit 10", conn)
surveyQ.head()
