import pandas as pd
df = pd.read_csv('extra/adult.csv')
print("Columns:", df.columns.tolist())
print("Unique education values:", df['education'].unique())
# Repeat for other columns as needed

df = pd.read_csv('extra/adult.csv')
edu_map = df[['education', 'educational-num']].drop_duplicates().set_index('education')['educational-num'].to_dict()
print(edu_map)