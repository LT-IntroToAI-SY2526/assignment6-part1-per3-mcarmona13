"""
Assignment 6 Part 1: Student Performance Prediction
Name: Max Carmona
Date: 11-20-25

This assignment predicts student test scores based on hours studied.
Complete all the functions below following the in-class ice cream example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the student scores data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """
    data = pd.read_csv(filename)

    print("=== Student Scores Data ===")
    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    return data


def create_scatter_plot(data):
    """
    Create a scatter plot to visualize the relationship between hours studied and scores
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data["Hours"], data["Scores"], color="purple", alpha=0.6)

    plt.xlabel("Hours Studied", fontsize=12)
    plt.ylabel("Test Score", fontsize=12)
    plt.title("Student Test Scores vs Hours Studied", fontsize=14, fontweight="bold")
    plt.grid(True, alpha=0.3)

    plt.savefig("scatter_plot.png", dpi=300, bbox_inches="tight")
    plt.show()

def split_data(data):
    """
    Split data into features (X) and target (y), then into training and testing sets
    
    Args:
        data: pandas DataFrame with Hours and Scores columns
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    X = data[["Hours"]]   # features
    y = data["Scores"]    # target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set:  {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    """
    Create and train a linear regression model
    
    Args:
        X_train: training features
        y_train: training target values
    
    Returns:
        trained LinearRegression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    slope = model.coef_[0]
    intercept = model.intercept_

    print("\n=== Model Training Complete ===")
    print(f"Slope (coefficient): {slope:.2f}")
    print(f"Intercept: {intercept:.2f}")
    print(f"Equation: Score = {slope:.2f} × Hours + {intercept:.2f}")

    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance on test data
    
    Args:
        model: trained LinearRegression model
        X_test: testing features
        y_test: testing target values
    Returns:
        predictions array
    """
    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\n=== Model Performance ===")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"Root Mean Squared Error: {rmse:.2f}")

    return predictions


def visualize_results(X_train, y_train, X_test, y_test, predictions, model):
    """
    Visualize the model's predictions against actual values
    
    Args:
        X_train: training features
        y_train: training target values
        X_test: testing features
        y_test: testing target values
        predictions: model predictions on test set
        model: trained model (to plot line of best fit)
    """
    plt.figure(figsize=(12, 6))

    plt.scatter(X_train, y_train, color="blue", alpha=0.5, label="Training Data")
    plt.scatter(X_test, y_test, color="green", alpha=0.7, label="Test Data (Actual)")
    plt.scatter(X_test, predictions, color="red", marker="x", s=100, label="Predictions")

    x_range = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    y_range = model.predict(x_range)

    plt.plot(x_range, y_range, color="black", linewidth=2, label="Line of Best Fit")

    plt.xlabel("Hours Studied", fontsize=12)
    plt.ylabel("Test Score", fontsize=12)
    plt.title("Linear Regression: Student Score Prediction", fontsize=14, fontweight="bold")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig("predictions_plot.png", dpi=300, bbox_inches="tight")
    plt.show()


def make_prediction(model, hours):
    """
    Make a prediction for a specific number of hours studied
    
    Args:
        model: trained LinearRegression model
        hours: number of hours to predict score for
    
    Returns:
        predicted test score
    """
    hours_array = np.array([[hours]])
    predicted_score = model.predict(hours_array)[0]

    print("\n=== New Prediction ===")
    print(f"If a student studies {hours} hours, predicted score is {predicted_score:.2f}")

    return predicted_score


if __name__ == "__main__":
    print("=" * 70)
    print("STUDENT PERFORMANCE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)
    
    # Step 1: Load and explore the data
    # TODO: Call load_and_explore_data() with 'student_scores.csv'
    
    # Step 2: Visualize the relationship
    # TODO: Call create_scatter_plot() with the data
    
    # Step 3: Split the data
    # TODO: Call split_data() and store the returned values
    
    # Step 4: Train the model
    # TODO: Call train_model() with training data
    
    # Step 5: Evaluate the model
    # TODO: Call evaluate_model() with the model and test data
    
    # Step 6: Visualize results
    # TODO: Call visualize_results() with all the necessary arguments
    
    # Step 7: Make a new prediction
    # TODO: Call make_prediction() for a student who studied 7 hours

    data = load_and_explore_data("student_scores.csv")
    create_scatter_plot(data)

    X_train, X_test, y_train, y_test = split_data(data)
    model = train_model(X_train, y_train)
    predictions = evaluate_model(model, X_test, y_test)
    visualize_results(X_train, y_train, X_test, y_test, predictions, model)

    make_prediction(model, 7)
    
    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part1_writeup.md!")
    print("=" * 70) 
