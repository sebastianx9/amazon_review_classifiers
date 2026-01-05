
import numpy as np
import matplotlib.pyplot as plt

def train_sentiment_classifier(M_train, sents_train, seed_value=42, num_features=5000, 
                               n_iters=2500, lr=0.1, lambda_l2=0.001, batch_size=128):
  
    np.random.seed(seed_value)
    y = np.array([int(l == "positive") for l in sents_train])
    weights = np.random.rand(num_features)
    bias = np.random.rand(1)
    logistic_loss = []
    num_samples = len(y)

    n_batches = num_samples // batch_size
    if num_samples % batch_size != 0:
        n_batches += 1

    for i in range(n_iters):
        shuffled_indices = np.random.permutation(num_samples)
        total_batch_loss = 0

        for batch_n in range(n_batches):
            start_index = batch_size * batch_n
            end_index = min((batch_size) * (batch_n + 1), num_samples)

            batch_indices = shuffled_indices[start_index:end_index]
            M_train_batch = M_train[batch_indices, :]
            y_train_batch = y[batch_indices]
            batch_samples = len(y_train_batch)

            z = M_train_batch.dot(weights) + bias
            q = 1 / (1 + np.exp(-z))
            eps = 0.00001
            Cross_Entropy_Loss = -sum((y_train_batch * np.log2(q + eps) + 
                                      (np.ones(len(y_train_batch)) - y_train_batch) * 
                                      np.log2(np.ones(len(y_train_batch)) - q + eps)))
            l2_penalty = lambda_l2 * np.sum(weights**2)
            batch_loss = Cross_Entropy_Loss + l2_penalty
            total_batch_loss += batch_loss

            dw = (q - y_train_batch).dot(M_train_batch) / batch_samples
            dw = dw + (2 * lambda_l2 * weights) / batch_samples
            db = sum(q - y_train_batch) / batch_samples
            weights = weights - lr * dw
            bias = bias - lr * db

        epoch_loss = total_batch_loss / n_batches
        logistic_loss.append(epoch_loss)

  
    plt.plot(range(1, n_iters), logistic_loss[1:])
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.title(f"Mini-batch Training With L2 Regularization (batch_size={batch_size})")
    plt.show()

    return weights, bias, logistic_loss
