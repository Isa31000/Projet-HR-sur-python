import pandas as pd

path='D:\\Documents\\VSCode\\Projet HR sur python\\HRDataset.csv'
raw_data=pd.read_csv(path)
data = pd.read_pickle("data_cleaned.pkl")
print(data.info())
