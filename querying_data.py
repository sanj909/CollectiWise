import pandas as pd
import numpy as np
import json
from google.cloud import bigquery


class SQL:
    def __init__(self, project = 'development-302415', dataset = 'machine_learning'):
        self.project = project
        self.client = bigquery.Client(project = self.project)
        self.dataset = self.client.dataset(dataset)
        self.query = []
        self.data = pd.DataFrame()
    
    def aggregate_to_intervals(self, interval_length, where = "WHERE RIGHT(asset,3) = 'USD'"):
        query = """
                #Length in seconds of each interval
                DECLARE
                interval_length INT64 DEFAULT """+str(interval_length)+""";

                #Adds a 'intervals' column which acts like an index for the interval each row belongs to
                CREATE OR REPLACE TABLE `development-302415.machine_learning.sorted_by_interval` AS
                WITH
                transactional AS (
                SELECT
                    *,
                    CAST(TRUNC(TIMESTAMP_DIFF(time_stamp,'2000-01-01 00:00:00+00', second)/interval_length,0) AS INT64) AS intervals,
                FROM
                    `development-302415.machine_learning.weekly_v1`
                """+where+"""
                ORDER BY
                    intervals,
                    asset )

                #Reverts 'intervals' index back to a timestamp, aggregates volume, average prices and OHLC prices over each interval (and over each asset)
                SELECT
                TIMESTAMP_ADD(TIMESTAMP '2000-01-01 00:00:00+00', INTERVAL t.intervals*interval_length second) AS time_stamp,
                t.asset,
                SUM(t.volume) AS volume,
                AVG(t.price) AS avg_price,
                AVG(open) AS open,
                MAX(t.price) AS high,
                MIN(t.price) AS low,
                AVG(close) AS close,
                AVG(label) as label,
                CASE
                    WHEN COUNT(t.price) >= 2 THEN STDDEV(t.price)
                ELSE
                0
                END
                AS std_price
                FROM (
                SELECT
                    *,
                    FIRST_VALUE(price) OVER(PARTITION BY intervals, asset ORDER BY time_stamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS open,
                    LAST_VALUE(price) OVER(PARTITION BY intervals, asset ORDER BY time_stamp ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS close,
                FROM
                    transactional) AS t
                GROUP BY
                t.intervals,
                t.asset
                ORDER BY t.intervals, label;
                """
        self.query.append(query)
        self.client.query(query)
        
    def convert_to_features(self, features = 'avg_price,volume,std_price,label'):
        query = """
                #DECLARE target_asset STRING DEFAULT 'ETHUSD';

                CALL `development-302415.machine_learning.pivot` (
                'development-302415.machine_learning.sorted_by_interval','development-302415.machine_learning.assets_to_features', ['time_stamp'], 'asset','[STRUCT("""+features+""")]',1000,'ANY_VALUE','');
        
        
                #CREATE OR REPLACE TABLE `development-302415.machine_learning.assets_to_features` AS
                #SELECT features.*, labels.label FROM `development-302415.machine_learning.assets_to_features` AS features
                #INNER JOIN `development-302415.machine_learning.sorted_by_interval` AS labels
                #ON features.time_stamp = labels.time_stamp AND labels.asset = target_asset
                #ORDER BY time_stamp;
                """
                    
        self.query.append(query)
        self.client.query(query)
        
    def get_table(self,table_name, max_results = None, csv_name = None):
        if max_results != None:
            self.data = self.client.query('SELECT * FROM `development-302415.'+table_name+'` LIMIT '+str(max_results))
        else:
            self.data = self.client.query('SELECT * FROM `development-302415.'+table_name+'`')
        self.data = self.data.result()
        self.data = self.data.to_dataframe()
        #table = self.client.get_table('development-302415.'+table_name)
        #self.data = self.client.list_rows(table).to_dataframe()
        if csv_name != None:
            self.data.to_csv(csv_name, index = False)
        return self.data

    def load_csv(self, path):
        self.data = pd.read_csv(path, header = 0, index_col = 0)
    
    def save_csv(self, path):
        self.data.to_csv(path)
         
    
    def unnest(self, columns_prefix = 'e_', na_fillers = {'avg_price':'ffill','volume':0,'std_price':0,'label':0,'high':'$avg_price','low':'$avg_price','open':'$avg_price','close':'$avg_price'}, dropna = False, merge_labels = True, label_name = 'label'):
        self.data = self.data.applymap(lambda x: x if x != '[]' else '[{}]')
        
        
        for column in self.data.columns.values:

            if column[:len(columns_prefix)] == columns_prefix:

                serie = self.data[column].map(lambda x: list(json.loads(x.replace('\'','\"')))[0])
                serie = pd.json_normalize(serie)
                serie.set_index(self.data.index,inplace = True)
                
                for feature in serie.columns.values:
                    try:
                        if type(na_fillers[feature]) == int:
                            serie[feature] = serie[feature].fillna(value = na_fillers[feature])
                            self.data[column+' '+feature] = serie[feature]
                        elif type(na_fillers[feature]) == str:
                            if na_fillers[feature][0] == '$':
                                serie[feature] = serie[feature].fillna(serie[na_fillers[feature][1:]])
                                self.data[column+' '+feature] = serie[feature]
                            else:
                                serie[feature] = serie[feature].fillna(method = na_fillers[feature])                      
                                self.data[column+' '+feature] = serie[feature]
                        else:
                            raise KeyError('Fill method isn\'t int or string for '+feature)

                    except KeyError:
                        raise KeyError('No NaN fill method declared for '+feature)

                self.data.drop(columns = column, inplace = True)
            
        #Puts all the '<asset> label' columns to the right
        if merge_labels:
            #print(self.data[[ i for i in list(self.data.columns) if i[-len(label_name):] == label_name]])
            labels = self.data[[ i for i in list(self.data.columns) if i[-len(label_name):] == label_name]].values.tolist()
                
            #print(labels)
            self.data.drop(columns = [ i for i in list(self.data.columns) if i[-len(label_name):] == label_name], inplace = True)
            self.data[label_name] = labels
        else:
            self.data = self.data[[ i for i in list(self.data.columns) if i[-len(label_name):] != label_name]+[ i for i in list(self.data.columns) if i[-len(label_name):] == label_name]]


        if dropna:
            self.data.dropna(axis = 0, inplace = True)

        return self.data
    
    def create_targets(self, targets = ['high','low'], merge_labels = True):
        for target in targets:
            df = self.data[[ i for i in list(self.data.columns) if i[-len(target):] == target]]
            df = df.rolling()
        
        
        
    def summarize(self, na_threshold = None):
        print('------------------------------------------------')
        if na_threshold != None:
            df = pd.Series(dtype = object)
            total = len(self.data.index.values)
            for column in self.data.columns.values:
                df[column] = self.data[column].isna().sum()/total*100

            df = df.where(df >= na_threshold).dropna().sort_values(ascending = False)

            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                print('Features with more than ',na_threshold,'% NaN: \n',df)
            print(len(df),' features with more than ',na_threshold,'% NaN values')
        
        print('Features: ',len(self.data.columns.values)-1,',\n'
            'Timestamps:',len(self.data.index.values))
        print('------------------------------------------------')

    def dropna(self, threshold = 100):

        df = pd.Series(dtype = object)
        total = len(self.data.index.values)
        for column in self.data.columns.values:
            df[column] = self.data[column].isna().sum()/total*100

        df = df.where(df >= threshold).dropna().sort_values(ascending = False)

        assets = [i[:-10] for i in df.index.values if i[-10:] == ' avg_price']
        self.data = self.data[[i for i in self.data.columns.values if list(filter(i.startswith, assets)) == []]]
        self.data.dropna(inplace = True, axis = 0) 

dir = 'cloudshell_open/CollectiWise/'
sql = SQL()

#sql.aggregate_to_intervals(7200)
sql.convert_to_features(features = 'avg_price,volume,std_price,label,high,low')
sql.get_table('machine_learning.assets_to_features', csv_name = dir+'assets_to_features.csv')
sql.data.drop(columns = ['time_stamp'], inplace=True)


sql.load_csv(dir+'assets_to_features.csv')
sql.unnest(merge_labels=True)
print(sql.data.columns.values)
sql.save_csv(dir+'features_df.csv')

sql.load_csv(dir+'features_df.csv')
#sql.summarize(na_threshold=50)
sql.dropna(threshold=50)
sql.summarize()
sql.data.set_index('label', inplace = True)
sql.save_csv(dir+'formatted_features.csv')

sql.load_csv(dir+'formatted_features.csv')
print(sql.data)
