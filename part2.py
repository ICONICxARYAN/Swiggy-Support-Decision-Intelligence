
import csv
import math
import os
import numpy as np

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("data/records.csv", "r") as file:
    rows = list(csv.DictReader(file))

X = np.array([
    [float(row["customer_tenure_months"]),
     float(row["earlier_ticket_count_90d"]),
     float(row["sla_consumed_ratio"]),
     float(row["account_value_inr_thousands"]),
     float(row["satisfaction_score"]),
     float(row["reopened_flag"]),
     float(row["self_service_first_flag"])]
    for row in rows
])

y = np.array([int(row["escalated"]) for row in rows])

results = {}

def save(key, value):
    results[key] = value

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def bce(actual, probability):
    probability = np.clip(probability, 1e-15, 1 - 1e-15)
    return -np.mean(
        actual * np.log(probability)
        + (1 - actual) * np.log(1 - probability)
    )

def confusion(actual, predicted):
    tp = int(np.sum((actual == 1) & (predicted == 1)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))
    return tp, fp, tn, fn

def metrics(tp, fp, tn, fn):
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = (2 * precision * recall / (precision + recall)
          if precision + recall else 0)
    return accuracy, precision, recall, f1, specificity

def write_results():
    with open("outputs/part2_results.txt", "w") as file:
        for key, value in results.items():
            if isinstance(value, (int, np.integer)):
                file.write(f"{key} = {value}\n")
            elif isinstance(value, str):
                file.write(f"{key} = {value}\n")
            else:
                file.write(f"{key} = {value:.4f}\n")


# --------------------------------------------------
# TASK 2.1: STRATIFIED 70/30 SPLIT
# --------------------------------------------------

zero_indices = np.where(y == 0)[0]
one_indices = np.where(y == 1)[0]

zero_cut = (7 * len(zero_indices)) // 10
one_cut = (7 * len(one_indices)) // 10

train_indices = np.sort(np.concatenate((
    zero_indices[:zero_cut],
    one_indices[:one_cut]
)))

test_indices = np.sort(np.concatenate((
    zero_indices[zero_cut:],
    one_indices[one_cut:]
)))

X_train_raw = X[train_indices]
X_test_raw = X[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

train_rows = len(train_indices)
test_rows = len(test_indices)

train_positives = int(np.sum(y_train))
test_positives = int(np.sum(y_test))

save("train_rows", train_rows)
save("train_positives", train_positives)
save("test_rows", test_rows)
save("test_positives", test_positives)

# Standardise using training data only
feature_mean = np.mean(X_train_raw, axis=0)
feature_sd = np.std(X_train_raw, axis=0, ddof=1)

X_train = (X_train_raw - feature_mean) / feature_sd
X_test = (X_test_raw - feature_mean) / feature_sd


# --------------------------------------------------
# TASK 2.2: LOGISTIC REGRESSION FROM SCRATCH
# --------------------------------------------------

def train_logistic(A, target, learning_rate, updates):
    number = len(target)
    weights = np.zeros(A.shape[1])
    bias = 0.0

    losses = {}

    probability = sigmoid(A @ weights + bias)
    losses[0] = bce(target, probability)

    for epoch in range(1, updates + 1):
        probability = sigmoid(A @ weights + bias)
        error = probability - target

        dw = (A.T @ error) / number
        db = np.sum(error) / number

        weights = weights - learning_rate * dw
        bias = bias - learning_rate * db

        if epoch in [1, 10, 100, 200, 1000]:
            probability = sigmoid(A @ weights + bias)
            losses[epoch] = bce(target, probability)

    return weights, bias, losses

weights, bias, losses = train_logistic(
    X_train, y_train, 0.1, 1000
)

save("loss_epoch_0", losses[0])
save("loss_epoch_1", losses[1])
save("loss_epoch_10", losses[10])
save("loss_epoch_100", losses[100])
save("loss_epoch_1000", losses[1000])

for i in range(7):
    save(f"weight_X{i + 1}", weights[i])

save("bias", bias)

test_probability = sigmoid(X_test @ weights + bias)
save("test_bce", bce(y_test, test_probability))


# --------------------------------------------------
# TASK 2.3: LEARNING RATE COMPARISON
# --------------------------------------------------

for eta in [0.01, 0.1, 1.0]:
    _, _, lr_losses = train_logistic(
        X_train, y_train, eta, 200
    )
    save(f"loss_epoch_200_eta_{eta}", lr_losses[200])


# --------------------------------------------------
# TASK 2.4: ODDS RATIOS
# --------------------------------------------------

for i in range(7):
    save(f"odds_ratio_X{i + 1}", math.exp(weights[i]))


# --------------------------------------------------
# TASK 2.5: TEST EVALUATION AT TWO THRESHOLDS
# --------------------------------------------------

for tau in [0.5, 0.3]:
    predicted = (test_probability >= tau).astype(int)

    tp, fp, tn, fn = confusion(y_test, predicted)

    accuracy, precision, recall, f1, specificity = metrics(
        tp, fp, tn, fn
    )

    prefix = f"tau_{tau}"

    save(prefix + "_tp", tp)
    save(prefix + "_fp", fp)
    save(prefix + "_tn", tn)
    save(prefix + "_fn", fn)
    save(prefix + "_accuracy", accuracy)
    save(prefix + "_precision", precision)
    save(prefix + "_recall", recall)
    save(prefix + "_f1", f1)
    save(prefix + "_specificity", specificity)

# Model predicting zero for every test record
blind_accuracy = np.mean(y_test == 0)
save("blind_accuracy", blind_accuracy)


# --------------------------------------------------
# TASK 2.6: THRESHOLD SWEEP / ROC DATA
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

with open("outputs/part2_thresholds.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"])

    for tau in np.arange(0.1, 1.0, 0.1):
        predicted = (test_probability >= tau).astype(int)

        tp, fp, tn, fn = confusion(y_test, predicted)

        tpr = tp / (tp + fn) if tp + fn else 0
        fpr = fp / (fp + tn) if fp + tn else 0

        writer.writerow([
            f"{tau:.4f}", tp, fp, tn, fn,
            f"{tpr:.4f}", f"{fpr:.4f}"
        ])


# --------------------------------------------------
# TASK 2.7: K-NEAREST NEIGHBOURS
# --------------------------------------------------

def knn_predict(reference_X, reference_y, query_X, k):
    predictions = []
    probabilities = []

    for query in query_X:
        distances = np.sqrt(
            np.sum((reference_X - query) ** 2, axis=1)
        )

        # Stable sorting preserves reference-row order for ties
        nearest = np.argsort(distances, kind="stable")[:k]

        positive_count = int(np.sum(reference_y[nearest]))
        probability = positive_count / k

        probabilities.append(probability)
        predictions.append(
            1 if positive_count >= (k + 1) // 2 else 0
        )

    return np.array(predictions), np.array(probabilities)


# Five-fold cross-validation
cv_results = []

for k in range(1, 22, 2):
    fold_f1_scores = []

    for fold in range(5):
        validation_positions = np.arange(train_rows) % 5 == fold
        reference_positions = ~validation_positions

        reference_X = X_train[reference_positions]
        reference_y = y_train[reference_positions]

        validation_X = X_train[validation_positions]
        validation_y = y_train[validation_positions]

        predictions, _ = knn_predict(
            reference_X, reference_y, validation_X, k
        )

        tp, fp, tn, fn = confusion(
            validation_y, predictions
        )

        _, _, _, f1, _ = metrics(tp, fp, tn, fn)

        fold_f1_scores.append(f1)

    mean_f1 = float(np.mean(fold_f1_scores))
    cv_results.append((k, mean_f1))

# Save CV results
with open("outputs/part2_knn_cv.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["K", "mean_cv_f1"])

    for k, mean_f1 in cv_results:
        writer.writerow([k, f"{mean_f1:.4f}"])

# Select highest F1; ties go to smaller K
knn_best_k = max(
    cv_results,
    key=lambda item: (item[1], -item[0])
)[0]

save("knn_best_k", knn_best_k)

# Refit using all training rows
knn_predictions, knn_probabilities = knn_predict(
    X_train, y_train, X_test, knn_best_k
)

knn_tp, knn_fp, knn_tn, knn_fn = confusion(
    y_test, knn_predictions
)

knn_accuracy, knn_precision, knn_recall, knn_f1, _ = metrics(
    knn_tp, knn_fp, knn_tn, knn_fn
)

save("knn_tp", knn_tp)
save("knn_fp", knn_fp)
save("knn_tn", knn_tn)
save("knn_fn", knn_fn)
save("knn_accuracy", knn_accuracy)
save("knn_precision", knn_precision)
save("knn_recall", knn_recall)
save("knn_f1", knn_f1)


# --------------------------------------------------
# TASK 2.8: COST-BASED THRESHOLD SELECTION
# --------------------------------------------------

best_tau = None
best_train_cost = None

train_probability = sigmoid(X_train @ weights + bias)

for k in range(1, 20):
    tau = k / 20

    train_prediction = (
        train_probability >= tau
    ).astype(int)

    tp, fp, tn, fn = confusion(y_train, train_prediction)

    cost = 50 * fp + 400 * fn

    # Strictly smaller keeps the smaller threshold in a tie
    if best_train_cost is None or cost < best_train_cost:
        best_train_cost = cost
        best_tau = tau

save("cost_tau", best_tau)
save("cost_train_at_tau", best_train_cost)

# Cost on test set at the selected training threshold
test_prediction_best = (
    test_probability >= best_tau
).astype(int)

tp, fp, tn, fn = confusion(y_test, test_prediction_best)

cost_test_at_tau = 50 * fp + 400 * fn
save("cost_test_at_tau", cost_test_at_tau)

# Cost at threshold 0.5
prediction_05 = (test_probability >= 0.5).astype(int)

tp, fp, tn, fn = confusion(y_test, prediction_05)

cost_test_at_05 = 50 * fp + 400 * fn
save("cost_test_at_0.5", cost_test_at_05)

# Cost when nobody is flagged
cost_test_flag_nobody = 400 * test_positives
save("cost_test_flag_nobody", cost_test_flag_nobody)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

write_results()

print("Part 2 completed successfully!")
print("Training rows:", train_rows)
print("Training positives:", train_positives)
print("Testing rows:", test_rows)
print("Testing positives:", test_positives)
print("Best K:", knn_best_k)
print("Best cost threshold:", best_tau)
print("Results saved in outputs/part2_results.txt")
print("CSV files saved successfully!")
