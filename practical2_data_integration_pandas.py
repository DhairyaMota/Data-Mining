
import pandas as pd

# Load datasets
df1 = pd.read_csv("Students1_pract2.csv")
df2 = pd.read_csv("Students2_pract2.csv")

print(df1.head())
print(df2.head())

# Merge datasets
df = pd.merge(df1, df2, on='Student Id')
print(df.head())

# Sorting by Age
sorted_df = df.sort_values(by=['Age'])
print(sorted_df)

# Filter specific columns
show = df.filter(['Age','Gender','Name'])
print(show)

# Find duplicates
print(df.duplicated())

# Remove duplicates
dups_removed = df.drop_duplicates()
print(dups_removed)

# Rename column
rename = df.rename(columns={'Student Id' : 'ID'})
print(rename)
