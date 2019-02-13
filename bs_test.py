import pandas as pd 
from package.utils.preprocess_text import processText 

data = pd.read_csv("D:\Research\TextClassification\MachineLearning\Q6\Q6_All.csv", encoding='latin-1', quoting=1)
# data.Q6_Text.apply(lambda x: str(x).encode('utf-8').decode('latin-1'))
print(data.head())
x = data.Q6_Text[2]
y = processText(x)
print(y)

data.Q6_Text = data.Q6_Text.apply(lambda x: processText(str(x), lemmatize=True))

print(data.Q6_Text.head())

print(data.Q6_Text[2])