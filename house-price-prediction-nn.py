import numpy as np
import csv


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
    feature_mapping = {'yes': 1, 'no': 0, 'furnished': 1, 'semi-furnished': 0.5, 'unfurnished': 0}
     
    with open(file, 'r') as file:
        csvreader = csv.reader(file)
        # Skip first line.
        next(csvreader)
        
        for row in csvreader:
            price = int(row[0])
            features = [feature_mapping[row[i]] if row[i] in feature_mapping else float(row[i]) for i in range(1, len(row))]
            X.append(features)
            prices.append(price)
        X = np.stack(X)
        y = np.array(prices)
        return X, y


def sigmoid(x, derv=False):
    if derv:
        return x * ( 1 - x)

    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A


class NeuralNetwork:
    def __init__(self, input_size: int=12) -> None:
        """
        """ 
        np.random.seed(42)
        self.w1 = np.random.rand(input_size, 64)
        self.b1 = np.zeros(64) 

        self.w2 = np.random.rand(64, 1)
        self.b2 = np.zeros(1)

    def train(self, X: np.array, y: np.array, epochs: int=10, learning_rate: float=0.001, verbose: bool=False) -> None:
        """
        """ 
        for epoch in range(epochs):
            # Forward propagation.
            Z1 = np.dot(X, self.w1) + self.b1
            A1 = sigmoid(Z1)
            Z2 = np.dot(A1, self.w2) + self.b2
            A2 = relu(Z2) 

            loss = np.mean((y.reshape(-1, 1) - A2) ** 2)
            
            # Backpropagation. 
            output_error = 2 * (y.reshape(-1, 1) - A2)
            hidden_error = np.dot(output_error, self.w2.T) * A2 * (1 - A2)
            
            self.w2 += learning_rate * np.dot(A2.T, output_error) / len(y)           
            self.b2 += learning_rate * np.sum(output_error) / len(y) 
            self.w1 += learning_rate * np.dot(X.T, hidden_error) / len(y)           
            self.b1 += learning_rate * np.sum(hidden_error) / len(y) 

            if verbose:
                if epoch % 1000 == 0:
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")


    def predict(self, X: np.array) -> int:
        # Forward propagation.
        Z1 = np.dot(X, self.w1) + self.b1
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, self.w2) + self.b2
        A2 = relu(Z2) 
        return A2

                

    def evaluate(self, X, y):
        predictions = self.predict(X)
        TP, TN, FP, FN = 0, 0, 0, 0
        for i in range(len(y)):
            if predictions[i] == 1 and y[i] == 1:
                TP += 1
            elif predictions[i] == 0 and  y[i] == 0:
                TN += 1
            elif predictions[i] == 1 and y[i] == 0:
                FP += 1
            else:
                FN += 1
        accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN != 0 else 0
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FP != 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
        return accuracy, precision, recall, f1


if __name__ == '__main__':
    X_train, y_train = read_labled_data('Housing.csv')
    nn = NeuralNetwork(12)
    nn.train(X_train, y_train, 100000, 0.1, True)
    accuracy, precision, recall, f1 = nn.evaluate(X_train, y_train)
    print()
    print(f"accuracy = {round(accuracy * 100, 2)}%")
    print(f"precision = {round(precision * 100, 2)}%")
    print(f"recall = {round(recall * 100, 2)}%")
    print(f"F1 = {round(f1 * 100, 2)}%")
