import numpy as np
import matplotlib.pyplot as plt


def multiplier_benefit(n_predictions: int):
    """Determines the added benefit of predicting against n number of trade pairs."""
    MINPAIRS = 0 # should never be touched
    MAXPAIRS = 5
    SOFTENING_COEF = 1.2 # lower is softer - more benefit to folks predicting on less terms - should alwasy be greater than 1
    BASE_BENEFIT = 0.7 # 30% of the benefit to people predicting only one term
    B = max(1, SOFTENING_COEF) # lower is softer - more benefit to folks predicting on less terms - should alwasy be greater than 1
    equivalent_pairs = np.clip(n_predictions, MINPAIRS, MAXPAIRS) # they'll get the same benefit as this many pair predictions
    if equivalent_pairs == MAXPAIRS:
        result = 1
    else:
        result = 1 + (BASE_BENEFIT * ((1 / B) ** (equivalent_pairs - 1))) # additional benefit
    return result


if __name__ == "__main__":
    # Define the range of x values - this will be the number of exchanges they're predicting against
    x = np.arange(0, 15)
    y = np.array([multiplier_benefit(n) for n in x])
    # z = a * (1 / b) ** 2
    # print(z)
    # Plot the curve
    plt.plot(x, y)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Exponential Curve with Decreasing Results for Bigger Values')
    plt.grid(True)
    plt.show()

