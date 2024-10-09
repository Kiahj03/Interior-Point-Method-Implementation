import pandas
import numpy

"""
Objective: Building a predictive model to forecast Tesla's future stock prices based on historical data and its selected features.

Approach: Applying the L1 regularized least squares algorithm to predict future prices. Regularization will help prevent overfitting and focus the model on important points and patterns.

Result: Model that highlights specific time periods or patterns that are most predictive of future prices.

Steps:
- Load data
- Create features and moving averages
- Define target
- Handle missing data
- Separate features and target
- Standardize features
- Implement Truncated Newton Interior Point Method
- Train model
- Predict with model
"""

# Load data
data = pandas.read_csv('TSLA.csv', parse_dates=['Date'])
data.sort_values('Date', inplace=True)
data.reset_index(drop=True, inplace=True)

# Create features and moving averages
data['Lag1'] = data['Close'].shift(1)
data['Lag2'] = data['Close'].shift(2)
data['MovingAvg5'] = data['Close'].rolling(window=5).mean()
data['MovingAvg10'] = data['Close'].rolling(window=10).mean()

# Define target
data['Target'] = data['Close'].shift(-1)

# Handle missing data
data = data.dropna()
data.reset_index(drop=True, inplace=True)

# Separate features and target
X = data[['Lag1', 'Lag2', 'MovingAvg5', 'MovingAvg10']].values
y = data['Target'].values
split = int(len(X) * 0.8)
XTrain, XTest = X[:split], X[split:]
yTrain, yTest = y[:split], y[split:]

# Standardize features
Xmean = XTrain.mean(axis=0)
Xstandard = XTrain.std(axis=0)
XTrainStandard = (XTrain - Xmean) / Xstandard
XTestStandard = (XTest - Xmean) / Xstandard

def add_intercept(X):
    intercept = numpy.ones((X.shape[0], 1))
    return numpy.hstack((intercept, X))
XTrainStandard = add_intercept(XTrainStandard)
XTestStandard = add_intercept(XTestStandard)

# Truncated Newton Interior Point Method
def TruncatedNewton(X, y, lamb, tol, maxIt):
    m, n = X.shape
    w = numpy.zeros(n)
    t = 1.0
    mu = 10.0 
    epsilon = 1e-8

    def f(w):
        residual = X.dot(w) - y
        return 0.5 * numpy.sum(residual ** 2) + lamb * numpy.sum(numpy.abs(w[1:]))

    def gradient(w):
        res = X.dot(w) - y
        grad = X.T.dot(res)
        grad[1:] += lamb * numpy.sign(w[1:])
        return grad

    def hessian():
        return numpy.dot(numpy.transpose(X), X)

    for _ in range(maxIt):
        grad = gradient(w)
        hess = hessian()
        
        try:
            delta= numpy.linalg.solve(hess + epsilon * numpy.eye(n), -grad)
        except numpy.linalg.LinAlgError:
            break

        alpha = 1.0
        beta = 0.5
        c = 1e-4

        while True:
            wNew= w + alpha * delta
            objectNew= f(wNew)
            objectTemp = f(w)
            if objectNew <= objectTemp + c * alpha * grad.dot(delta):
                break
            alpha *= beta

        w = wNew

        if numpy.linalg.norm(grad, ord=2) < tol:
            break
        t *= mu
    return w

# Train model
lamb = 1
tol = 1e-6
maxIt = 100
weights = TruncatedNewton(XTrainStandard, yTrain, lamb, tol, maxIt)
print(f'Optimized weights: {weights}') #Output: [207.86546873  14.32859197 -17.94814265  50.56671876 -14.15862499]

# Predict with model
lastR = data[['Lag1', 'Lag2', 'MovingAvg5', 'MovingAvg10']].iloc[-1].values
lastR = (lastR - Xmean) / Xstandard
lastR = numpy.insert(lastR, 0, 1)
prediction = lastR.dot(weights)
print(f'Predicted next day price: {prediction} USD') #Output: 221.5763324596194 
