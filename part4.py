
import csv
import math
import os
import numpy as np

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# LOAD DATA AND RECREATE PART 2 SPLIT
# --------------------------------------------------

with open("data/records.csv", "r") as file:
    rows = list(csv.DictReader(file))

X_original = np.array([
    [
        float(row["customer_tenure_months"]),
        float(row["earlier_ticket_count_90d"]),
        float(row["sla_consumed_ratio"]),
        float(row["account_value_inr_thousands"]),
        float(row["satisfaction_score"]),
        float(row["reopened_flag"]),
        float(row["self_service_first_flag"])
    ]
    for row in rows
])

y = np.array([int(row["escalated"]) for row in rows])

zero_indices = np.where(y == 0)[0]
one_indices = np.where(y == 1)[0]

train_indices = np.sort(np.concatenate([
    zero_indices[:7 * len(zero_indices) // 10],
    one_indices[:7 * len(one_indices) // 10]
]))

test_indices = np.setdiff1d(
    np.arange(len(y)), train_indices
)

X_train_original = X_original[train_indices]
X_test_original = X_original[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

mean = np.mean(X_train_original, axis=0)
sd = np.std(X_train_original, axis=0, ddof=1)

X_train = (X_train_original - mean) / sd
X_test = (X_test_original - mean) / sd

n_train = len(y_train)

results = {}


def save(key, value):
    results[key] = value


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def log_loss(y_true, probability):
    probability = np.clip(probability, 1e-15, 1 - 1e-15)

    return -np.mean(
        y_true * np.log(probability)
        + (1 - y_true) * np.log(1 - probability)
    )


def metrics(y_true, prediction):
    tp = int(np.sum((y_true == 1) & (prediction == 1)))
    fp = int(np.sum((y_true == 0) & (prediction == 1)))
    tn = int(np.sum((y_true == 0) & (prediction == 0)))
    fn = int(np.sum((y_true == 1) & (prediction == 0)))

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return tp, fp, tn, fn, precision, recall, f1


# --------------------------------------------------
# RECREATE PART 2 LOGISTIC REGRESSION
# --------------------------------------------------

weights = np.zeros(7)
bias = 0.0
eta = 0.1

for epoch in range(1001):
    probability = sigmoid(X_train @ weights + bias)

    if epoch in [0, 1, 10, 100, 1000]:
        save(f"logistic_loss_epoch_{epoch}",
             log_loss(y_train, probability))

    if epoch < 1000:
        error = probability - y_train

        weights -= eta * (X_train.T @ error) / n_train
        bias -= eta * np.mean(error)

logistic_train_probability = sigmoid(
    X_train @ weights + bias
)

logistic_test_probability = sigmoid(
    X_test @ weights + bias
)

save("logistic_test_bce",
     log_loss(y_test, logistic_test_probability))


# --------------------------------------------------
# RECREATE PART 2 KNN
# --------------------------------------------------

def knn_probability(reference_X, reference_y, query_X, k):
    predictions = []

    for query in query_X:
        distances = np.sqrt(
            np.sum((reference_X - query) ** 2, axis=1)
        )

        nearest = np.argsort(
            distances, kind="stable"
        )[:k]

        predictions.append(
            np.mean(reference_y[nearest])
        )

    return np.array(predictions)


knn_k_values = list(range(1, 22, 2))
knn_scores = {}

for k in knn_k_values:
    fold_scores = []

    for fold in range(5):
        validation_mask = (
            np.arange(n_train) % 5 == fold
        )

        reference_X = X_train[~validation_mask]
        reference_y = y_train[~validation_mask]

        validation_X = X_train[validation_mask]
        validation_y = y_train[validation_mask]

        probabilities = knn_probability(
            reference_X,
            reference_y,
            validation_X,
            k
        )

        predictions = (
            probabilities >= (k + 1) / (2 * k)
        ).astype(int)

        fold_scores.append(
            metrics(validation_y, predictions)[-1]
        )

    knn_scores[k] = np.mean(fold_scores)

best_k = max(
    knn_scores,
    key=lambda k: (knn_scores[k], -k)
)

save("knn_best_k", best_k)

knn_test_probability = knn_probability(
    X_train, y_train, X_test, best_k
)

knn_test_prediction = (
    knn_test_probability >= (best_k + 1) / (2 * best_k)
).astype(int)


# --------------------------------------------------
# TASK 4.1: ENTROPY, GINI AND INFORMATION GAIN
# --------------------------------------------------

def entropy(labels):
    if len(labels) == 0:
        return 0.0

    p = np.mean(labels)

    if p == 0 or p == 1:
        return 0.0

    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def gini(labels):
    if len(labels) == 0:
        return 0.0

    p = np.mean(labels)

    return 1 - p ** 2 - (1 - p) ** 2


root_entropy = entropy(y_train)
root_gini = gini(y_train)

save("root_entropy", root_entropy)
save("root_gini", root_gini)

splits = [
    ("S1", X_train_original[:, 5] == 1),
    ("S2", X_train_original[:, 6] == 1),
    ("S3", X_train_original[:, 2] > 0.5),
    ("S4", X_train_original[:, 0] <= 24)
]

split_results = []

for name, condition in splits:
    yes = y_train[condition]
    no = y_train[~condition]

    weighted_entropy = (
        len(yes) / n_train * entropy(yes)
        + len(no) / n_train * entropy(no)
    )

    weighted_gini = (
        len(yes) / n_train * gini(yes)
        + len(no) / n_train * gini(no)
    )

    info_gain = root_entropy - weighted_entropy
    gini_gain = root_gini - weighted_gini

    split_results.append([
        name,
        len(yes),
        int(np.sum(yes)),
        len(no),
        int(np.sum(no)),
        info_gain,
        gini_gain
    ])

best_info_index = max(
    range(4),
    key=lambda i: split_results[i][5]
)

best_gini_index = max(
    range(4),
    key=lambda i: split_results[i][6]
)

best_split = splits[best_info_index][1]
best_split_name = split_results[best_info_index][0]

save("best_split_by_info_gain",
     split_results[best_info_index][0])

save("best_split_by_gini_gain",
     split_results[best_gini_index][0])

stump_rate_yes = np.mean(y_train[best_split])
stump_rate_no = np.mean(y_train[~best_split])

save("stump_rate_yes", stump_rate_yes)
save("stump_rate_no", stump_rate_no)


# --------------------------------------------------
# TASK 4.2: STUMP, VOTING AND MODEL COMPARISON
# --------------------------------------------------

stump_test_side = (
    X_test_original[:, splits[best_info_index][0] == "S1"] 
    if False else None
)

# Apply the selected split condition to test rows
stump_conditions = [
    X_test_original[:, 5] == 1,
    X_test_original[:, 6] == 1,
    X_test_original[:, 2] > 0.5,
    X_test_original[:, 0] <= 24
]

stump_test_side = stump_conditions[best_info_index]

stump_probability = np.where(
    stump_test_side,
    stump_rate_yes,
    stump_rate_no
)

stump_prediction = (
    stump_probability >= 0.3
).astype(int)

logistic_prediction = (
    logistic_test_probability >= 0.3
).astype(int)

hard_vote = (
    logistic_prediction
    + knn_test_prediction
    + stump_prediction
    >= 2
).astype(int)

soft_vote = (
    (logistic_test_probability
     + knn_test_probability
     + stump_probability) / 3
    >= 0.3
).astype(int)

model_predictions = [
    ("logistic", logistic_prediction),
    ("knn", knn_test_prediction),
    ("stump", stump_prediction),
    ("hard_vote", hard_vote),
    ("soft_vote", soft_vote)
]

voting_rows = []

for name, prediction in model_predictions:
    values = metrics(y_test, prediction)

    voting_rows.append([name, *values])


# --------------------------------------------------
# TASK 4.3: ADABOOST BY HAND
# --------------------------------------------------

true_y = np.array([
    1, 1, 1, -1, -1, -1, 1, -1
])

h1 = np.array([
    1, 1, -1, -1, -1, -1, -1, -1
])

h2 = np.array([
    -1, 1, 1, -1, 1, -1, 1, -1
])


def adaboost_round(weights, predictions):
    error = np.sum(
        weights[predictions != true_y]
    )

    alpha = 0.5 * math.log(
        (1 - error) / error
    )

    unnormalised = weights * np.exp(
        -alpha * true_y * predictions
    )

    updated = unnormalised / np.sum(unnormalised)

    return error, alpha, unnormalised, updated


initial_weights = np.ones(8) / 8

error1, alpha1, unnormalised1, weights1 = adaboost_round(
    initial_weights, h1
)

error2, alpha2, unnormalised2, weights2 = adaboost_round(
    weights1, h2
)

save("round1_error", error1)
save("round1_alpha", alpha1)
save("round2_error", error2)
save("round2_alpha", alpha2)

ensemble_prediction = np.sign(
    alpha1 * h1 + alpha2 * h2
)

ensemble_accuracy = np.mean(
    ensemble_prediction == true_y
)

save("ensemble_accuracy", ensemble_accuracy)


# --------------------------------------------------
# TASK 4.4: OUT-OF-BAG FRACTION
# --------------------------------------------------

oob_fraction = (1 - 1 / 560) ** 560

save("oob_fraction", oob_fraction)


# --------------------------------------------------
# TASK 4.5: NEURAL NETWORK ARCHITECTURE
# --------------------------------------------------

# Model A: Fully connected network

mlp_hidden1 = 784 * 16 + 16
mlp_hidden2 = 16 * 16 + 16
mlp_output = 16 * 10 + 10

mlp_total = (
    mlp_hidden1 + mlp_hidden2 + mlp_output
)

# Model A: Convolutional network

# Input: 28 x 28 x 1
# Conv1: 28 x 28 x 6
# Pool1: 14 x 14 x 6
# Conv2: 10 x 10 x 16
# Pool2: 5 x 5 x 16
# Flatten: 400

cnn_conv1 = (5 * 5 * 1 + 1) * 6
cnn_conv2 = (5 * 5 * 6 + 1) * 16
cnn_dense1 = 400 * 32 + 32
cnn_dense2 = 32 * 10 + 10

cnn_total = (
    cnn_conv1 + cnn_conv2 + cnn_dense1 + cnn_dense2
)

# Fully connected versus convolutional scaling

fc_layer_params = 128 * 128 * 3 * 1024 + 1024

conv_layer_params = (
    3 * 3 * 3 * 64 + 64
)

# Model B: Recurrent neural network

W_xh = (128, 64)
W_hh = (128, 128)
W_hy = (16, 128)

rnn_total = (
    128 * 64
    + 128 * 128
    + 16 * 128
    + 128
    + 16
)

save("mlp_total", mlp_total)
save("cnn_total", cnn_total)
save("fc_layer_params", fc_layer_params)
save("conv_layer_params", conv_layer_params)
save("rnn_total", rnn_total)

save("W_xh_shape", "128x64")
save("W_hh_shape", "128x128")
save("W_hy_shape", "16x128")


# --------------------------------------------------
# SAVE PART 4 RESULTS
# --------------------------------------------------

with open("outputs/part4_results.txt", "w") as file:
    for key, value in results.items():
        if isinstance(value, str):
            file.write(f"{key} = {value}\n")
        else:
            file.write(f"{key} = {value:.4f}\n")


# Splits CSV
with open("outputs/part4_splits.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "split", "n_yes", "positives_yes",
        "n_no", "positives_no",
        "info_gain", "gini_gain"
    ])

    for row in split_results:
        writer.writerow([
            row[0], row[1], row[2], row[3], row[4],
            f"{row[5]:.4f}", f"{row[6]:.4f}"
        ])


# Voting CSV
with open("outputs/part4_voting.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "model", "tp", "fp", "tn", "fn",
        "precision", "recall", "f1"
    ])

    for row in voting_rows:
        writer.writerow([
            row[0], *row[1:5],
            *[f"{v:.4f}" for v in row[5:]]
        ])


# AdaBoost CSV
with open("outputs/part4_adaboost.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "round", "point", "weight_before",
        "weight_unnormalised", "weight_after"
    ])

    for i in range(8):
        writer.writerow([
            1, i + 1,
            f"{initial_weights[i]:.6f}",
            f"{unnormalised1[i]:.6f}",
            f"{weights1[i]:.6f}"
        ])

    for i in range(8):
        writer.writerow([
            2, i + 1,
            f"{weights1[i]:.6f}",
            f"{unnormalised2[i]:.6f}",
            f"{weights2[i]:.6f}"
        ])


# Architecture CSV
with open("outputs/part4_architecture.csv", "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow(["layer", "parameters"])

    writer.writerows([
        ["mlp_hidden1", mlp_hidden1],
        ["mlp_hidden2", mlp_hidden2],
        ["mlp_output", mlp_output],
        ["cnn_conv1", cnn_conv1],
        ["cnn_conv2", cnn_conv2],
        ["cnn_dense1", cnn_dense1],
        ["cnn_dense2", cnn_dense2]
    ])


print("Part 4 completed successfully!")
print("Training rows:", len(y_train))
print("Testing rows:", len(y_test))
print("Best information gain split:", best_split_name)
print("Best K:", best_k)
print("Round 1 error:", round(error1, 4))
print("Out-of-bag fraction:", round(oob_fraction, 4))
print("MLP parameters:", mlp_total)
print("CNN parameters:", cnn_total)
print("RNN parameters:", rnn_total)
print("Results saved in outputs/part4_results.txt")
print("CSV files saved successfully!")
