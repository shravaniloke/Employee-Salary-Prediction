import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from pickle import dump

# Load dataset
data = pd.read_csv("Salary.csv")
print(data.head())

# Feature and target
x = data[['YearsExperience']]
y = data['Salary']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# Model performance
r2 = model.score(x_test, y_test)
print("R² score: ", r2)

# -------- Visualization --------

# Sort values for smooth line
x_sorted = data['YearsExperience'].sort_values()
'''
Takes YearsExperience column
Sorts it from small to large
Example:
Before:  [1.3, 5.0, 2.1]
After:   [1.3, 2.1, 5.0]
If X is not sorted, the line plot looks zig-zag instead of straight.
'''
x_sorted_df = x_sorted.values.reshape(-1, 1)

'''
model.predict() -> expects 2D input
.values         -> converts Series to array
.reshape(-1, 1) -> converts to column shape
[1.3, 2.1, 5.0]   ❌
[[1.3], [2.1], [5.0]] ✅
'''

# Predict
y_pred = model.predict(x_sorted_df)
'''
Uses trained model
Predicts salary for each experience value
These points form the regression line
'''

# Plot
plt.figure(figsize=(10, 6))   #set width:10inches height:6inches

plt.scatter(data['YearsExperience'], data['Salary'], color='red', label='Actual data')
#set red dots , each dot = real data point

#Line plot
plt.plot(x_sorted, y_pred, color='blue', label='Regression line')
#This plots blue straight line (Linear Regression line)

plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience (Linear Regression)')
plt.legend()
plt.show()

#generate a pickle file (load)
f = open("sp.pkl", "wb")
dump(model, f)
f.close()
print("model saved")