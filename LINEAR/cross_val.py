import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import confusion_matrix, classification_report
data = pd.read_csv("sales_data.csv")
x = data[['Advertising Budget', 'Store Size', 'Location Score']]

y_reg = data['Monthly Sales']
y_clf = (data['Monthly Sales'] >= 25000).astype(int)

x_train, x_test, y_reg_train, y_reg_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    x, y_clf, test_size=0.2, random_state=42
)

lin_model = LinearRegression()
lin_model.fit(x_train, y_reg_train)

y_reg_pred = lin_model.predict(x_test)

print("Linear Regression Results")
print("MSE:", mean_squared_error(y_reg_test, y_reg_pred))
print("R2 Score:", r2_score(y_reg_test, y_reg_pred))


log_model = LogisticRegression(max_iter=1000)
log_model.fit(x_train, y_clf_train)

y_clf_pred = log_model.predict(x_test)

print("\nLogistic Regression Results")
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test, y_clf_pred))

print("\nClassification Report:")
print(classification_report(y_clf_test, y_clf_pred))


knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_clf_train)

y_knn_pred = knn_model.predict(x_test)

print("\nKNN Results")
print("Confusion Matrix:")
print(confusion_matrix(y_clf_test, y_knn_pred))

print("\nClassification Report:")
print(classification_report(y_clf_test, y_knn_pred))


print("\nCross Validation Scores (Logistic Regression):")
cv_scores = cross_val_score(log_model, x, y_clf, cv=5)

print("Scores:", cv_scores)
print("Average Score:", cv_scores.mean())