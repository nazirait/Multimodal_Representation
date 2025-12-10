import pandas as pd
from pathlib import Path

print("="*20, "AMAZON", "="*20)
df_train = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/amazon/train.csv")
df_test = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/amazon/test.csv")
def_val = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/amazon/val.csv")

print("Train label counts:\n", df_train['label'].value_counts(), "\n")
print("Test label counts:\n", df_test['label'].value_counts(), "\n")
print("Validation set label counts:\n", def_val['label'].value_counts(), "\n")
print("Train missing values:\n", df_train.isnull().sum(), "\n")
print("Test missing values:\n", df_test.isnull().sum(), "\n")
print("Validation set missing values:\n", def_val.isnull().sum(), "\n")


print("="*20, "FASHION", "="*20)
df_fashion_train = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/fashionai/train.csv")
df_fashion_val = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/fashionai/val.csv")

print("Fashion Train label counts:\n", df_fashion_train['label'].value_counts(), "\n")
print("Fashion Val label counts:\n", df_fashion_val['label'].value_counts(), "\n")
print("Fashion Train missing values:\n", df_fashion_train.isnull().sum(), "\n")
print("Fashion Val missing values:\n", df_fashion_val.isnull().sum(), "\n")

print("="*20, "MOVIELENS", "="*20)
df_ml_train = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/movielens/train.csv")
df_ml_val = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/movielens/val.csv")
df_ml_test = pd.read_csv("D:/COmparative_Study_of_Multimodal_Represenations/data/processed/movielens/test.csv")

print("Genres (train, top 10):\n", df_ml_train['genres'].value_counts().head(10), "\n")
print("Movielens Train missing values:\n", df_ml_train.isnull().sum(), "\n")
print("Movielens Val missing values:\n", df_ml_val.isnull().sum(), "\n")
print("Movielens Test missing values:\n", df_ml_test.isnull().sum(), "\n")
