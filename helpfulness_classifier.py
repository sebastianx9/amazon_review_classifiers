
import numpy as np
import matplotlib.pyplot as plt

def train_helpfulness_classifier(M_train, y_train, num_features=5000, num_classes=3,
                                 seed_value=42, n_iters=2500, lr=0.05,
                                 lambda_l2=0.001, batch_size=128):
 

    np.random.seed(seed_value)
    num_samples = len(y_train)
    weights = np.random.rand(num_classes, num_features)
    bias = np.random.rand(num_classes, 1)
    logistic_loss = []

    z = np.zeros((num_samples, num_classes))
    q = np.zeros((num_samples, num_classes))

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
            y_train_batch = y_train_batch = y_train[batch_indices, :]
            batch_samples = len(y_train_batch)

            z = M_train_batch.dot(weights.T) + bias.T  # (batch_size, num_classes)
            z_sum = np.exp(z).sum(axis=1)
            q = np.array([list(np.exp(z_i) / z_sum[i]) for i, z_i in enumerate(z)])  # (batch_size, num_classes)
            l2_penalty = lambda_l2 * np.sum(weights**2)
            Cross_Entropy_loss = np.mean(-np.log2((np.sum((y_train_batch * q), axis=1))))  # (batch_size,)
            batch_loss = Cross_Entropy_loss + l2_penalty
            total_batch_loss += batch_loss

            dw = (q.T - y_train_batch.T).dot(M_train_batch) / batch_samples + 2 * lambda_l2 * weights  # (num_classes, num_features)
            db = (q.T - y_train_batch.T).sum(axis=1, keepdims=True) / batch_samples
            weights = (weights - (dw * lr))
            bias = bias - (db * lr)

        epoch_loss = total_batch_loss / n_batches
        logistic_loss.append(epoch_loss)


    plt.plot(range(1, n_iters), logistic_loss[1:])
    plt.xlabel("number of epochs")
    plt.ylabel("loss")
    plt.title(f"Mini-batch Training With L2 Regularization (batch_size={batch_size})")
    plt.show()

    return weights, bias, logistic_loss