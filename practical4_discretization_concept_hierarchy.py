
import pandas as pd

df = pd.read_csv("market_basket.csv")
print(df.head())

# Discretization
price_bins = [0,1,2,5,10]
price_labels = ['Low','Medium','High','Very High']

df['Price_Category'] = pd.cut(df['Price'], bins=price_bins, labels=price_labels, right=False)

# Concept Hierarchy
concept_hierarchy = {
'Apple':'Fruit',
'Banana':'Fruit',
'Orange':'Fruit',
'Milk':'Dairy',
'Bread':'Grain',
'Butter':'Dairy',
'Shampoo':'Toiletries',
'Toothpaste':'Toiletries',
'Chicken':'Meat',
'Beef':'Meat'
}

df['Concept_Hierarchy'] = df['Product'].map(concept_hierarchy)

print("\nDataset After Discretization and Concept Hierarchy:")
print(df.head())
