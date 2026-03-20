import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("sales_data.csv")

# Features
x = data[['Advertising Budget', 'Store Size', 'Location Score']]

# Targets
y_reg = data['Monthly Sales']
y_clf = (data['Monthly Sales'] >= 25000).astype(int)

# Split data
x_train, x_test, y_reg_train, y_reg_test = train_test_split(
    x, y_reg, test_size=0.2, random_state=42
)

_, _, y_clf_train, y_clf_test = train_test_split(
    x, y_clf, test_size=0.2, random_state=42
)

# Linear Regression
lin_model = LinearRegression()
lin_model.fit(x_train, y_reg_train)

y_reg_pred = lin_model.predict(x_test)

print("Linear Regression Predictions:")
print(y_reg_pred[:5])

print("MSE:", mean_squared_error(y_reg_test, y_reg_pred))
print("R2 Score:", r2_score(y_reg_test, y_reg_pred))

# Logistic Regression
log_model = LogisticRegression()
log_model.fit(x_train, y_clf_train)

y_clf_pred = log_model.predict(x_test)

print("\nLogistic Regression Predictions:")
print(y_clf_pred[:5])

print("MSE:", mean_squared_error(y_clf_test, y_clf_pred))
print("R2 Score:", r2_score(y_clf_test, y_clf_pred))