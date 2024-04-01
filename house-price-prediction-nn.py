import numpy as np
import csv
import pandas as pd
from sklearn.preprocessing import StandardScaler


"""
price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,
hotwaterheating,airconditioning,parking,prefarea,furnishingstatus
"""


def read_labled_data(file: str) -> tuple[np.array, np.array]:
    """
    >>> X, y = read_labled_data('housing-doctest.csv')
    >>> X
    array([[7.42e+03, 4.00e+00, 2.00e+00, 3.00e+00, 1.00e+00, 0.00e+00,
            0.00e+00, 0.00e+00, 1.00e+00, 2.00e+00, 1.00e+00, 1.00e+00],
           [8.96e+03, 4.00e+00, 4.00e+00, 4.00e+00, 1.00e+00, 0.00e+00,
            0.00e+00, 0.00e+00, 1.00e+00, 3.00e+00, 0.00e+00, 1.00e+00],
           [9.96e+03, 3.00e+00, 2.00e+00, 2.00e+00, 1.00e+00, 0.00e+00,
            1.00e+00, 0.00e+00, 0.00e+00, 2.00e+00, 1.00e+00, 5.00e-01]])
    >>> y
    array([13300000, 12250000, 12250000])
    """ 
    X = [] # Inputs (area, bedrooms, bathrooms...).
    prices = []
    
    # Mapping dictionary for categorical features
    feature_mapping = {'yes': 1, 'no': 0, 'furnished': 1, 'semi-furnished': 0, 'unfurnished': -1}
     
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        # Skip first line.
        next(csvreader)
        
        for row in csvreader:
            price = int(row[0])
            features = [float(feature_mapping[row[i]]) if row[i] in feature_mapping else float(row[i]) for i in range(1, len(row))]
            X.append(features)
            prices.append(price)
        X = np.array(X)
        y = np.array(prices)
        return X, y

def read_data(file: str) -> tuple[np.array, np.array]:
    boston = pd.read_csv(file, header=None, delimiter=r"\s+", names=['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'])
    X = boston.iloc[:,:-1]
    y = boston.iloc[:, -1]
    X = np.array(X)
    y = np.array(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    return X, y


def sigmoid(x: np.array) -> np.array:
    return 1 / (1 + np.exp(-x))

def ReLu(x):
    return np.maximum(0, x)

class NeuralNetwork:
    def __init__(self, input_size: int=12) -> None:
        """
        """ 
        self.w1 = np.zeros((input_size, 4))
        self.b1 = np.zeros(4)

        self.w2 = np.zeros((4, 1))
        self.b2 = np.zeros(1)

    def train(self, X: np.array, y: np.array, epochs: int=10, learning_rate: float=0.01, verbose: bool=False) -> None:
        """
        """ 
        for epoch in range(epochs):
            # Forward propagation.
            Z1 = np.dot(X, self.w1) + self.b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, self.w2) + self.b2
            A2 = Z2 # Predictions.
            
            loss = np.mean((A2 - y.reshape(-1, 1)) ** 2)

            # Backpropagation. 
            output_error = 2 * (A2 - y.reshape(-1, 1))
            hidden_error = np.dot(output_error, self.w2.T) * A1 * (1 - A1)

            self.w2 -= learning_rate * np.dot(A1.T, output_error) / len(y)
            self.b2 -= learning_rate * np.sum(output_error) / len(y)
            self.w1 -= learning_rate * np.dot(X.T, hidden_error) / len(y)
            self.b1 -= learning_rate * np.sum(hidden_error) / len(y)

            if verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
                    # print(f"{self.w1}")
                    # print()
                    # print(f"{self.w2}")

    def predict(self, X: np.array) -> int:
        # Forward propagation.
        Z1 = np.dot(X, self.w1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        A2 = Z2 
        return A2.ravel()

                
    def evaluate(self, X, y):
        predictions = self.predict(X)
        mse = np.mean((predictions - y) ** 2)
        return mse


if __name__ == '__main__':
    X_train, y_train = read_data('housing2.csv')
    nn = NeuralNetwork(13)
    nn.train(X_train, y_train, 5000, 0.001, True)
    loss = nn.evaluate(X_train, y_train)
    print(loss)
