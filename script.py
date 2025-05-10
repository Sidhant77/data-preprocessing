import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df= pd.read_csv("C:/Users/Sidhant/Git Hub/all_files/FILES.csv")
x=df.iloc[:,0:4]
y=df["salary"]
sc=StandardScaler()
x_final=sc.fit_transform(x)
x_train,x_test,y_train,y_test=train_test_split(x_final,y,test_size=0.2,random_state=42)


print(y_train)