import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

df = pd.read_parquet("features/faces/affectnet_openface.parquet")
drop_cols = ["frame","face_id","timestamp","success","label"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
clf_lr = LogisticRegression(max_iter=2000)
clf_lr.fit(X_train, y_train)
print("Logistic Regression Test Accuracy:", clf_lr.score(X_test, y_test))
print(classification_report(y_test, clf_lr.predict(X_test)))

# Random Forest
clf_rf = RandomForestClassifier(n_estimators=200, random_state=42)
clf_rf.fit(X_train, y_train)
print("Random Forest Test Accuracy:", clf_rf.score(X_test, y_test))
print(classification_report(y_test, clf_rf.predict(X_test)))
