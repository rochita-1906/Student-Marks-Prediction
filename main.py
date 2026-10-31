
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# Load the dataset
data = pd.read_csv("student_scores.csv")

# Separate input and output
X = data[["Hours"]]
y = data["Marks"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Test the model
predictions = model.predict(X_test)

print("\nSTUDENT MARKS PREDICTION")
print("------------------------")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")
print(f"R-squared Score: {r2_score(y_test, predictions):.2f}")

# Predict marks based on user input
while True:
    user_input = input("\nEnter study hours (or type 'q' to quit): ")

    if user_input.lower() == "q":
        print("Thank you for using Student Marks Prediction!")
        break

    try:
        hours = float(user_input)

        if hours < 0:
            print("Please enter a positive number.")
            continue

        new_data = pd.DataFrame({"Hours": [hours]})
        predicted_marks = model.predict(new_data)[0]

        predicted_marks = max(0, min(100, predicted_marks))

        print(f"Predicted marks for {hours:g} study hours: {predicted_marks:.2f}/100")

    except ValueError:
        print("Please enter a valid number.")
