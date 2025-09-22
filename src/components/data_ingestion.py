#this will play a important role in data ingestion
import os
import sys
import src.exception as CustomException
import logging
import src.logger  # This ensures your custom logger configuration is applied
import pandas as pd
#from src.utils import load_object
from sklearn.model_selection import train_test_split
from dataclasses import dataclass #this has started maybe from python 3.9, it will help us to create a class without init method

@dataclass
#we can just call the class and pass the variables without init method or other methods
class DataIngestionConfig: #this class will help us to store the path of train, test and raw data
    train_data_path: str=os.path.join('artifacts','train.csv') #artifacts is a folder where we will store all the files related to our project
    test_data_path: str=os.path.join('artifacts','test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig() #this will create an object of DataIngestionConfig class and we can access the variables of that class using this object, the values would be saved

    def initiate_data_ingestion(self):
        logging.info('Data Ingestion method starts')
        try:
            df=pd.read_csv('notebook/data/stud.csv') #reading the data
            logging.info('Dataset read as pandas dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #this will create the artifacts folder if it does not exist

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True) #this will save the raw data in the artifacts folder
            logging.info('Raw data is saved')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=2) #splitting the data into train and test set

            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) #this will save the train data in the artifacts folder
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) #this will save the test data in the artifacts folder

            logging.info('Ingestion of data is completed')

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)

        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()