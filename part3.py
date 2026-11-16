
import csv
import math
import os
import numpy as np

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

with open("data/records.csv", "r") as file:
    rows = list(csv.DictReader(file))

# Five continuous predictors
X_original = np.array([
    [float(row["customer_tenure_months"]),
     float(row["earlier_ticket_count_90d"]),
     float(row["sla_consumed_ratio"]),
     float(row["account_value_inr_thousands"]),
     float(row["satisfaction_score"])]
    for row in rows
])

outcome = np.array([int(row["escalated"]) for row in rows])
value = np.array([float(row["resolution_minutes"]) for row in rows])

n = len(rows)
results = {}

def save(key, value):
    results[key] = value

def write_results():
    with open("outputs/part3_results.txt", "w") as file:
        for key, value in results.items():
            if isinstance(value, (int, np.integer)):
                file.write(f"{key} = {value}\n")
            elif isinstance(value, str):
                file.write(f"{key} = {value}\n")
            else:
                file.write(f"{key} = {value:.4f}\n")


# --------------------------------------------------
# TASK 3.1: STANDARDISATION
# --------------------------------------------------

feature_mean = np.mean(X_original, axis=0)
feature_sd = np.std(X_original, axis=0, ddof=1)

X = (X_original - feature_mean) / feature_sd


# --------------------------------------------------
# TASK 3.1: K-MEANS FROM SCRATCH
# --------------------------------------------------

def kmeans(data, k):
    number = len(data)

    step = number // k

    # Pinned starting centroid rows
    centroids = data[
        np.array([i * step for i in range(k)])
    ].copy()

    previous_labels = np.full(number, -1)
    passes = 0

    while True:
        passes += 1

        # Calculate squared Euclidean distances
        distances = np.sum(
            (data[:, np.newaxis, :] -
             centroids[np.newaxis, :, :]) ** 2,
            axis=2
        )

        # argmin chooses the lower centroid number in a tie
        labels = np.argmin(distances, axis=1)

        # Stop when no assignment changes
        if np.array_equal(labels, previous_labels):
            break

        previous_labels = labels.copy()

        # Update centroids
        for j in range(k):
            members = data[labels == j]

            if len(members) > 0:
                centroids[j] = np.mean(members, axis=0)

    # Calculate final WCSS
    wcss = 0.0

    for j in range(k):
        members = data[labels == j]

        if len(members) > 0:
            wcss += np.sum((members - centroids[j]) ** 2)

    return labels, centroids, wcss, passes


# Run K-means for K = 2 to 6
wcss_values = {}

for k in range(2, 7):
    labels, centroids, wcss, passes = kmeans(X, k)

    wcss_values[k] = wcss

    save(f"wcss_k{k}", wcss)


# --------------------------------------------------
# TASK 3.2: ELBOW METHOD
# --------------------------------------------------

elbow_k = 5

for k in [3, 4, 5]:
    previous_drop = wcss_values[k - 1] - wcss_values[k]
    current_drop = wcss_values[k] - wcss_values[k + 1]

    if current_drop < 0.5 * previous_drop:
        elbow_k = k
        break

save("elbow_k", elbow_k)


# --------------------------------------------------
# TASK 3.3: K-MEANS WITH K = 3
# --------------------------------------------------

labels, centroids, wcss3, passes3 = kmeans(X, 3)

# Order clusters by ascending mean of original X3
cluster_information = []

for cluster_number in range(3):
    indices = np.where(labels == cluster_number)[0]

    mean_x3 = np.mean(X_original[indices, 2])

    cluster_information.append(
        (mean_x3, cluster_number, indices)
    )

cluster_information.sort(key=lambda item: item[0])

segment_indices = {}

with open("outputs/part3_segments.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "segment", "size", "mean_X1", "mean_X2",
        "mean_X3", "mean_X4", "mean_X5",
        "outcome_rate", "mean_value"
    ])

    for position, item in enumerate(cluster_information):
        mean_x3, cluster_number, indices = item

        segment = chr(ord("A") + position)

        segment_indices[segment] = indices

        segment_size = len(indices)

        save(f"size_segment_{segment}", segment_size)

        feature_values = np.mean(
            X_original[indices], axis=0
        )

        outcome_rate = np.mean(outcome[indices])
        mean_value = np.mean(value[indices])

        writer.writerow([
            segment,
            segment_size,
            *[f"{v:.4f}" for v in feature_values],
            f"{outcome_rate:.4f}",
            f"{mean_value:.4f}"
        ])


# --------------------------------------------------
# TASK 3.4: HIERARCHICAL CLUSTERING
# --------------------------------------------------

distance_matrix = np.array([
    [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
])

def cluster_distance(cluster_a, cluster_b, linkage):
    distances = [
        distance_matrix[i][j]
        for i in cluster_a
        for j in cluster_b
    ]

    if linkage == "single":
        return min(distances)

    elif linkage == "complete":
        return max(distances)

    else:
        return sum(distances) / len(distances)


def cluster_name(cluster):
    return "+".join(
        f"E{i + 1}" for i in sorted(cluster)
    )


def hierarchical(linkage):
    clusters = [[i] for i in range(6)]

    merge_heights = []
    merge_records = []

    for merge_number in range(1, 6):

        best_distance = float("inf")
        best_pair = None

        # Find the two closest clusters
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):

                distance = cluster_distance(
                    clusters[i],
                    clusters[j],
                    linkage
                )

                if distance < best_distance:
                    best_distance = distance
                    best_pair = (i, j)

        i, j = best_pair

        cluster_a = clusters[i]
        cluster_b = clusters[j]

        # Order clusters by smallest member number
        if min(cluster_a) > min(cluster_b):
            cluster_a, cluster_b = cluster_b, cluster_a

        merge_records.append([
            linkage,
            merge_number,
            cluster_name(cluster_a),
            cluster_name(cluster_b),
            best_distance
        ])

        merge_heights.append(best_distance)

        merged = sorted(cluster_a + cluster_b)

        # Remove old clusters and add merged cluster
        clusters = [
            cluster for index, cluster in enumerate(clusters)
            if index not in (i, j)
        ]

        clusters.append(merged)

        clusters.sort(key=lambda cluster: min(cluster))

    # Find the largest consecutive merge-height gap
    gaps = [
        merge_heights[i + 1] - merge_heights[i]
        for i in range(4)
    ]

    largest_gap_index = int(np.argmax(gaps))

    merges_below_cut = largest_gap_index + 1

    # Rebuild clusters at the cut
    cut_clusters = [[i] for i in range(6)]

    for merge_number in range(merges_below_cut):

        target = merge_records[merge_number]

        members_a = [
            int(x[1:]) - 1
            for x in target[2].split("+")
        ]

        members_b = [
            int(x[1:]) - 1
            for x in target[3].split("+")
        ]

        new_cluster = sorted(members_a + members_b)

        cut_clusters = [
            cluster for cluster in cut_clusters
            if cluster != members_a and cluster != members_b
        ]

        cut_clusters.append(new_cluster)

    cut_clusters.sort(key=lambda cluster: min(cluster))

    cut_text = ";".join(
        cluster_name(cluster) for cluster in cut_clusters
    )

    return merge_records, cut_text, len(cut_clusters)


hierarchical_records = []

for linkage in ["single", "complete", "average"]:

    records_for_linkage, cut_text, count = hierarchical(linkage)

    hierarchical_records.extend(records_for_linkage)

    save(f"{linkage}_clusters_at_cut", cut_text)

# Save all 15 hierarchical merges
with open("outputs/part3_hierarchical.csv", "w", newline="") as file:

    writer = csv.writer(file)

    writer.writerow([
        "linkage", "merge_no", "cluster_a",
        "cluster_b", "height"
    ])

    for record in hierarchical_records:
        writer.writerow([
            record[0],
            record[1],
            record[2],
            record[3],
            f"{record[4]:.4f}"
        ])


# --------------------------------------------------
# TASK 3.5: PRINCIPAL COMPONENT ANALYSIS
# --------------------------------------------------

# Step 1: Mean vector of standardised data
pca_mean = np.mean(X, axis=0)

# Step 2: Centre the data
X_centered = X - pca_mean

# Step 3: Covariance matrix
covariance = (X_centered.T @ X_centered) / (n - 1)

# Step 4: Eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eigh(covariance)

# Step 5: Sort largest eigenvalue first
order = np.argsort(eigenvalues)[::-1]

eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

# Step 6: Apply the sign convention
for j in range(5):

    largest_index = np.argmax(
        np.abs(eigenvectors[:, j])
    )

    if eigenvectors[largest_index, j] < 0:
        eigenvectors[:, j] *= -1

# Step 7: Keep top two components
W = eigenvectors[:, :2]

projections = X_centered @ W

# Eigenvalue calculations
eigenvalue_sum = np.sum(eigenvalues)

save("eigenvalue_sum", eigenvalue_sum)

cumulative = 0.0
components_for_80pct = 5

for i in range(5):

    ratio = eigenvalues[i] / eigenvalue_sum

    cumulative += ratio

    save(f"eigenvalue_{i + 1}", eigenvalues[i])
    save(f"explained_ratio_{i + 1}", ratio)
    save(f"cumulative_ratio_{i + 1}", cumulative)

    if cumulative >= 0.8 and components_for_80pct == 5:
        components_for_80pct = i + 1

save("components_for_80pct", components_for_80pct)

# PC1 and PC2 loadings
for i in range(5):
    save(f"pc1_loading_X{i + 1}", W[i, 0])
    save(f"pc2_loading_X{i + 1}", W[i, 1])

# Projections of the first three records
for i in range(3):
    save(f"row{i + 1}_pc1", projections[i, 0])
    save(f"row{i + 1}_pc2", projections[i, 1])

# Mean PC1 score of each segment
for segment, indices in segment_indices.items():

    mean_pc1 = np.mean(projections[indices, 0])

    save(f"mean_pc1_segment_{segment}", mean_pc1)


# --------------------------------------------------
# SAVE FINAL RESULTS
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

write_results()

print("Part 3 completed successfully!")
print("Total records:", n)
print("K-means WCSS for K=2:", round(wcss_values[2], 4))
print("Elbow K:", elbow_k)
print("Segment sizes:",
      [len(segment_indices[s]) for s in ["A", "B", "C"]])
print("PCA eigenvalue sum:", round(eigenvalue_sum, 4))
print("Results saved in outputs/part3_results.txt")
print("CSV files saved successfully!")
