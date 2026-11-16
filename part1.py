
import csv
import math
import numpy as np

# Load dataset
with open("data/records.csv", "r") as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# Extract required columns
x4 = np.array([float(row["account_value_inr_thousands"]) for row in rows])
y = np.array([float(row["resolution_minutes"]) for row in rows])

X1 = np.array([float(row["customer_tenure_months"]) for row in rows])
X2 = np.array([float(row["earlier_ticket_count_90d"]) for row in rows])
X3 = np.array([float(row["sla_consumed_ratio"]) for row in rows])
X5 = np.array([float(row["satisfaction_score"]) for row in rows])
X6 = np.array([float(row["reopened_flag"]) for row in rows])
X7 = np.array([float(row["self_service_first_flag"]) for row in rows])

n = len(y)
results = {}

def save(key, value):
    results[key] = value

def write_results():
    with open("outputs/part1_results.txt", "w") as file:
        for key, value in results.items():
            if isinstance(value, (int, np.integer)):
                file.write(f"{key} = {value}\n")
            elif isinstance(value, str):
                file.write(f"{key} = {value}\n")
            else:
                file.write(f"{key} = {value:.4f}\n")


# --------------------------------------------------
# TASK 1.1: ESTIMATION AND CONFIDENCE INTERVAL
# --------------------------------------------------

mean_value = np.mean(y)
sd_value = np.std(y, ddof=1)
se_value = sd_value / math.sqrt(n)

ci95_low = mean_value - 1.96 * se_value
ci95_high = mean_value + 1.96 * se_value

save("n", n)
save("mean_value", mean_value)
save("sd_value", sd_value)
save("se_value", se_value)
save("ci95_low", ci95_low)
save("ci95_high", ci95_high)


# --------------------------------------------------
# TASK 1.2: HYPOTHESIS TESTING
# --------------------------------------------------

z_stat = (mean_value - 8.0) / se_value
reject_h0 = "yes" if abs(z_stat) > 1.96 else "no"

save("z_stat", z_stat)
save("reject_h0", reject_h0)


# --------------------------------------------------
# TASK 1.3: REQUIRED SAMPLE SIZE
# --------------------------------------------------

n_required = math.ceil((1.96 * sd_value / 0.5) ** 2)

save("n_required", n_required)


# --------------------------------------------------
# TASK 1.4: SIMPLE LINEAR REGRESSION
# --------------------------------------------------

x_mean = np.mean(x4)
y_mean = np.mean(y)

reg_slope = np.sum((x4 - x_mean) * (y - y_mean)) / np.sum((x4 - x_mean) ** 2)

reg_intercept = y_mean - reg_slope * x_mean

fitted = reg_intercept + reg_slope * x4
residuals = y - fitted

tss = np.sum((y - y_mean) ** 2)
ess = np.sum((fitted - y_mean) ** 2)
rss = np.sum(residuals ** 2)

tss_equals_ess_plus_rss = "yes" if abs(tss - ess - rss) < 1e-6 * tss else "no"

r2 = 1 - rss / tss
rmse = math.sqrt(rss / n)
mae = np.mean(np.abs(residuals))

save("reg_slope", reg_slope)
save("reg_intercept", reg_intercept)
save("tss", tss)
save("ess", ess)
save("rss", rss)
save("tss_equals_ess_plus_rss", tss_equals_ess_plus_rss)
save("r2", r2)
save("rmse", rmse)
save("mae", mae)


# --------------------------------------------------
# TASK 1.5: GRADIENT DESCENT
# --------------------------------------------------

x_sd = np.std(x4, ddof=1)
z = (x4 - x_mean) / x_sd

m = 0.0
c = 0.0
eta = 0.1

def calculate_cost(m, c):
    error = y - (m * z + c)
    return np.mean(error ** 2)

save("gd_J0", calculate_cost(m, c))

for i in range(1, 501):
    prediction = m * z + c
    error = y - prediction

    dm = -(2 / n) * np.sum(z * error)
    dc = -(2 / n) * np.sum(error)

    m = m - eta * dm
    c = c - eta * dc

    if i in [1, 10, 100, 500]:
        save(f"gd_J{i}", calculate_cost(m, c))

gd_slope = m / x_sd
gd_intercept = c - m * x_mean / x_sd
gd_slope_gap = abs(gd_slope - reg_slope)

save("gd_slope", gd_slope)
save("gd_intercept", gd_intercept)
save("gd_slope_gap", gd_slope_gap)


# --------------------------------------------------
# TASK 1.6: RESIDUAL ANALYSIS AND LOG REGRESSION
# --------------------------------------------------

order = np.argsort(fitted, kind="stable")

lower_indices = order[:400]
upper_indices = order[400:]

resid_sd_lower_raw = np.std(residuals[lower_indices], ddof=1)
resid_sd_upper_raw = np.std(residuals[upper_indices], ddof=1)

resid_sd_ratio_raw = resid_sd_upper_raw / resid_sd_lower_raw

save("resid_sd_lower_raw", resid_sd_lower_raw)
save("resid_sd_upper_raw", resid_sd_upper_raw)
save("resid_sd_ratio_raw", resid_sd_ratio_raw)

# Log-log regression
log_x = np.log(x4)
log_y = np.log(y)

log_x_mean = np.mean(log_x)
log_y_mean = np.mean(log_y)

loglog_slope = np.sum((log_x - log_x_mean) * (log_y - log_y_mean)) / np.sum((log_x - log_x_mean) ** 2)

loglog_intercept = log_y_mean - loglog_slope * log_x_mean

log_fitted = loglog_intercept + loglog_slope * log_x
log_residuals = log_y - log_fitted

log_tss = np.sum((log_y - log_y_mean) ** 2)
log_rss = np.sum(log_residuals ** 2)

loglog_r2 = 1 - log_rss / log_tss
loglog_rmse = math.sqrt(log_rss / n)

log_order = np.argsort(log_fitted, kind="stable")

log_lower = log_order[:400]
log_upper = log_order[400:]

resid_sd_lower_log = np.std(log_residuals[log_lower], ddof=1)
resid_sd_upper_log = np.std(log_residuals[log_upper], ddof=1)

resid_sd_ratio_log = resid_sd_upper_log / resid_sd_lower_log

save("loglog_intercept", loglog_intercept)
save("loglog_slope", loglog_slope)
save("loglog_r2", loglog_r2)
save("loglog_rmse", loglog_rmse)
save("resid_sd_lower_log", resid_sd_lower_log)
save("resid_sd_upper_log", resid_sd_upper_log)
save("resid_sd_ratio_log", resid_sd_ratio_log)


# --------------------------------------------------
# TASK 1.7: MULTIPLE LINEAR REGRESSION
# --------------------------------------------------

X = np.column_stack((
    np.ones(n),
    log_x,
    X1,
    X2,
    X3,
    X5,
    X6,
    X7
))

beta = np.linalg.solve(X.T @ X, X.T @ log_y)

multiple_fitted = X @ beta
multiple_residuals = log_y - multiple_fitted

multiple_tss = np.sum((log_y - np.mean(log_y)) ** 2)
multiple_rss = np.sum(multiple_residuals ** 2)

mr_r2 = 1 - multiple_rss / multiple_tss

mr_adj_r2 = 1 - (1 - mr_r2) * (n - 1) / (n - 7 - 1)

loglog_adj_r2 = 1 - (1 - loglog_r2) * (n - 1) / (n - 1 - 1)

for i in range(8):
    save(f"mr_beta_{i}", beta[i])

save("mr_r2", mr_r2)
save("mr_adj_r2", mr_adj_r2)
save("loglog_adj_r2", loglog_adj_r2)


# --------------------------------------------------
# SAVE FINAL RESULTS
# --------------------------------------------------

write_results()

print("Part 1 completed successfully!")
print("Total records:", n)
print("Mean resolution time:", round(mean_value, 4))
print("Regression R2:", round(r2, 4))
print("Gradient descent slope gap:", round(gd_slope_gap, 6))
print("Results saved in outputs/part1_results.txt")
