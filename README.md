# EX-02-Cross-Platform-Prompting-Evaluating-Diverse-Techniques-in-AI-Powered-Text-Summarization

## AIM
Exploratory data analysis using python

##Algorithm:
1.Import necessary libraries
2.Load and understand the dataset
3.Clean and preprocess the data
4.Perform statistical analysis and visualization
5.Interpret insights and conclude

## PROGRAM AND OUTPUT:
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv")
print("First 5 rows:")
display(df.head())
~~~
<img width="741" height="255" alt="image" src="https://github.com/user-attachments/assets/9d7fe0c4-a8a2-48c9-9d89-6924445fb8cb" />
~~~
print("\nDataset Info:")
df.info()
~~~
<img width="529" height="522" alt="image" src="https://github.com/user-attachments/assets/778514bb-1d17-4a13-b36d-1fd1320975fe" />
~~~
print("\nStatistical Summary:")
display(df.describe())
~~~
<img width="686" height="355" alt="image" src="https://github.com/user-attachments/assets/469cd643-8319-4745-b3ff-ae75ff37eb4c" />
~~~
print("\nMissing Values:")
print(df.isnull().sum())
~~~
<img width="263" height="396" alt="image" src="https://github.com/user-attachments/assets/63480245-fa2f-417c-9bb8-0fe80dfecd38" />
~~~
plt.figure()
df['age'].hist(bins=20)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()
~~~
<img width="726" height="575" alt="image" src="https://github.com/user-attachments/assets/596b31e6-31f5-414c-b0a6-724bd488173a" />
~~~
plt.figure()
sns.countplot(x='survived', data=df)
plt.title("Survival Count")
plt.show()
~~~
<img width="825" height="413" alt="image" src="https://github.com/user-attachments/assets/a29d260b-33f2-41e2-bb8f-b5dd35992dbd" />
~~~
plt.figure()
sns.boxplot(x='survived', y='age', data=df)
plt.title("Age vs Survival")
plt.show()
~~~
<img width="766" height="600" alt="image" src="https://github.com/user-attachments/assets/9284341c-c5af-4fa4-9e52-40a64e7042c8" />
~~~
plt.figure()
sns.scatterplot(x='age', y='fare', data=df)
plt.title("Age vs Fare")
plt.show()
~~~
<img width="748" height="560" alt="image" src="https://github.com/user-attachments/assets/72153be6-ac3c-4d8a-8871-342ae6a63b63" />
~~~
sns.pairplot(df[['age', 'fare', 'survived']])
plt.show()
~~~
<img width="743" height="746" alt="image" src="https://github.com/user-attachments/assets/2d0eb4c9-0bf9-4b73-a756-f42c406be269" />
~~~
plt.figure()
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True)
plt.title("Correlation Matrix")
plt.show()
print("\nEDA Completed Successfully!")
~~~
<img width="788" height="691" alt="image" src="https://github.com/user-attachments/assets/6c730d5e-da8a-4504-bf39-4e1fdefaae5a" />

## RESULT
Exploratory Data analysis performed in jupiter notebook successfully
