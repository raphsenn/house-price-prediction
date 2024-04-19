import numpy as np
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(self, input_neurons: int=13, hidden_neurons: int=8, output_neurons: int=1) -> None:
        """
        >>> NN = NeuralNetwork(13, 4, 1)
        >>> NN.w1
        array([[0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.],
               [0., 0., 0., 0.]])
        >>> NN.w2
        array([[0.],
               [0.],
               [0.],
               [0.]])
        """ 
        self.w1 = np.zeros((input_neurons, hidden_neurons))     # 13 x 4
        self.b1 = np.zeros(hidden_neurons)                      # 4 x 1

        self.w2 = np.zeros((hidden_neurons, output_neurons))    # 4 x 1
        self.b2 = np.zeros(output_neurons)                      # 1 x 1

    def sigmoid(self, x: np.array, derv:bool=False) -> np.array:
        if derv:
            return self.sigmoid(x) * (1 - self.sigmoid(x))
        return 1.0 / (1.0 + np.exp(-x))
    
    def relu(self, x: np.array, derv:bool=False) -> np.array:
        if derv:
            return x > 0
        return np.maximum(0, x)

    def train(self, X: np.array, y: np.array, epochs: int=5000, learning_rate: float=0.001, verbose: bool=False) -> None:
        for epoch in range(epochs):
            # Forward propagation.
            Z1 = np.dot(X, self.w1) + self.b1                   # 405 x 1
            A1 = self.sigmoid(Z1)                               # 405 x 1
            Z2 = np.dot(A1, self.w2) + self.b2                  # 1 x 405
            A2 = Z2 # Predictions.                              # 1 x 405

            # Calculate loss. 
            loss = np.mean((A2 - y.reshape(-1, 1)) ** 2)

            # Backpropagation. 
            output_error = 2 * (A2 - y.reshape(-1, 1))
            hidden_error = np.dot(output_error, self.w2.T) * self.sigmoid(Z1, derv=True)

            self.w2 -= learning_rate * np.dot(A1.T, output_error) / len(y)
            self.b2 -= learning_rate * np.sum(output_error) / len(y)
            self.w1 -= learning_rate * np.dot(X.T, hidden_error) / len(y)
            self.b1 -= learning_rate * np.sum(hidden_error) / len(y)

            if verbose:
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X: np.array) -> np.array:
        # Forward propagation.
        Z1 = np.dot(X, self.w1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        A2 = Z2
        return A2

    def evaluate(self, X: np.array, y: np.array) -> float:
        predictions = self.predict(X).ravel()
        mse = np.mean((predictions - y) ** 2)
        return mse


def read_data(file: str) -> tuple[np.array, np.array]:
    """
    >>> X, y = read_labled_data('housing-doctest.csv')
    >>> X
    array([[-1.41421308,  1.41421356, -1.41421356,  0.        ,  1.41421356,
            -0.46074429, -0.42044647, -1.41421356, -1.41421356,  1.41421356,
            -1.41421356,  0.70710678, -0.48217443],
           [ 0.70811766, -0.70710678,  0.70710678,  0.        , -0.70710678,
            -0.92755101,  1.37958998,  0.70710678,  0.70710678, -0.70710678,
             0.70710678,  0.70710678,  1.39244766],
           [ 0.70609543, -0.70710678,  0.70710678,  0.        , -0.70710678,
             1.3882953 , -0.95914351,  0.70710678,  0.70710678, -0.70710678,
             0.70710678, -1.41421356, -0.91027322]])
    >>> y
    array([24. , 21.6, 34.7])
    """ 
    X = []
    prices = []
    with open(file, 'r') as file:
        for line in file:
            data = line.split()
            features = [float(data[i]) for i in range(0, len(data) - 1)]
            price = float(data[13])
            X.append(features)
            prices.append(price)
    X = np.array(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = np.array(prices)
    return X, y


if __name__ == '__main__':
    X, y = read_data('tests/housing.csv')

    # Split for training and testing data.
    X_train, y_train = X[:405], y[:405] # training data.
    X_test, y_test = X[405:], y[405:] # testing data.

    # Create a NeuralNet. 
    nn = NeuralNetwork(13, 8, 1)
    # Train the NeuralNet.
    nn.train(X_train, y_train, 5000, 0.001, False)
    # Evaluate on testing data. 
    MSE = nn.evaluate(X_test, y_test)
    RMSE = np.sqrt(MSE) 
    print(f"Root mean squared error: {round(RMSE, 2)}")
