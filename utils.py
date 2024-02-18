import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats


# for test.ipynb and test(2, 3, 4).ipynb

# Define the function to return the SMAPE value
def smape(actual, predicted) -> float:
  
    # Convert actual and predicted to numpy
    # array data type if not already
    if not all([isinstance(actual, np.ndarray), 
                isinstance(predicted, np.ndarray)]):
        actual, predicted = np.array(actual),
        np.array(predicted)
  
    return round(
        np.mean(
            np.abs(predicted - actual) / 
            ((np.abs(predicted) + np.abs(actual))/2)
        )*100, 6
    )


def plot_eval(pred, true):
    residuals = true - pred

    # Create a 2x3 grid of subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Scatter plot of real vs. predicted values
    axes[0, 0].scatter(true, pred)
    axes[0, 0].set_xlabel('Real Values')
    axes[0, 0].set_ylabel('Predicted Values')
    axes[0, 0].set_title('Scatter Plot')

    # Residual plot
    axes[0, 1].scatter(pred, residuals)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Values')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residual Plot')

    # Histogram of residuals
    axes[0, 2].hist(residuals, bins=20)
    axes[0, 2].set_xlabel('Residuals')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Histogram of Residuals')

    # QQ plot
    stats.probplot(residuals.flatten(), plot=axes[1, 0])
    axes[1, 0].set_title('QQ Plot')

    # Regression line plot
    axes[1, 1].scatter(true, pred)
    axes[1, 1].plot(true, true, color='r', linestyle='--')
    axes[1, 1].set_xlabel('Real Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Regression Line Plot')

    # R-squared and MSE values
    r2 = r2_score(true, pred)
    mse = mean_squared_error(true, pred)
    axes[1, 2].bar(['R-squared', 'MSE'], [r2, mse])
    axes[1, 2].set_title('R-squared and MSE')

    # Adjust layout and show plot
    plt.tight_layout()
    plt.show()