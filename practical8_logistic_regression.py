
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
import pandas as pd

data = {
'Age':[22,25,47,52,46,56,55,60,62,61,18,28],
'Income':[50000,55000,65000,70000,75000,80000,60000,70000,75000,55000,40000,45000],
'Gender':[0,1,0,1,0,1,0,1,0,1,0,1],
'Purchased':[0,1,1,1,0,1,0,1,1,1,0,0]
}

df = pd.DataFrame(data)

X = df[['Age','Income','Gender']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

selector = SelectKBest(chi2, k='all')
X_new = selector.fit_transform(X_train_scaled, y_train)

logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_new, y_train)

y_pred = logreg.predict(X_test)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Classification Report:")
print(classification_report(y_test,y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test,y_pred))
