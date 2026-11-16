Domain: Support

# Swiggy Support Decision Intelligence

## Project Overview

This project analyses synthetic Swiggy support ticket records using statistical methods and machine learning algorithms.

The project focuses on understanding resolution time, predicting ticket escalations, identifying customer support segments and evaluating different predictive models.

The dataset contains 800 synthetic records and 10 columns.

## Part 1 findings

### R1.1 Hypothesis Testing

The calculated mean resolution time is 8.9382 minutes, compared with the assumed 8.0 minutes.

The Z-statistic is 2.8930, which is greater than the critical value of 1.96. Therefore, we reject the null hypothesis at the 5% significance level.

A Type I error would mean concluding that the actual average resolution time differs from 8 minutes when it is actually 8 minutes.

The difference is 0.9382 minutes, approximately 11.7% above the assumed average. This difference could matter for staffing and capacity planning when applied across a large number of support tickets.

### R1.2 Sample Size

The required sample size is 1293 records, whereas the available dataset contains 800 records.

This means the current dataset is smaller than the sample required to estimate the average resolution time within a margin of error of 0.5 minutes at 95% confidence.

The planning team should consider collecting more records for a more precise estimate.

### R1.3 Gradient Descent

The gradient descent slope gap is 0.0000.

Gradient descent reached the same regression line as the closed-form method because it repeatedly updated the slope and intercept using the calculated gradients until convergence.

### R1.4 Residual Analysis

The raw residual standard deviation ratio is 3.7525, indicating a funnel pattern where residual spread increases with fitted values.

After applying the log transformation, the ratio decreased to 1.0396.

This shows that the log transformation substantially reduces the funnel pattern and produces a more consistent residual spread.

### R1.5 Multiple Regression

The predictors with absolute coefficients greater than 0.05, excluding the constant, are:

- ln(X4): Account value
- X2: Earlier ticket count in 90 days
- X3: SLA consumed ratio
- X6: Reopened flag

The multiple regression adjusted R-squared is 0.7403, compared with 0.6561 for the log-log regression.

The improvement of 0.0842 indicates that the additional predictors improve the model's explanatory performance. The coefficient threshold is a mechanical selection rule, not a ranking of predictor importance.

## Part 2 findings

### R2.1 Learning Rate Comparison

The three learning-rate losses from lowest to highest are:

- η = 1.0: 0.3703
- η = 0.1: 0.3776
- η = 0.01: 0.5091

The smaller learning rate of 0.01 learns more slowly within 200 updates, while the larger learning rate of 1.0 achieves the lowest loss in this experiment without destabilising training.

### R2.2 Important Predictors

The three predictors with the largest absolute logistic regression weights are:

- X3: SLA consumed ratio (coefficient 0.9013)
- X1: Customer tenure months (coefficient -0.5554)
- X2: Earlier ticket count in 90 days (coefficient 0.5505)

The largest odds ratio is 2.4629 for X3 (SLA consumed ratio). This means that a one-training-standard-deviation increase in SLA consumed ratio multiplies the estimated odds of ticket escalation by approximately 2.46, keeping other predictors fixed.

### R2.3 Classification Threshold Comparison

The accuracy of the model predicting zero for every record is 0.7875, compared with 0.8042 at the logistic regression threshold of 0.5.

Accuracy alone is a weak measure because the dataset has fewer escalated tickets than non-escalated tickets, so a model can achieve reasonably high accuracy while missing many escalations.

When the threshold is reduced from 0.5 to 0.3:

- Recall increases from 0.4706 to 0.7451.
- Precision decreases from 0.5455 to 0.4634.

This means more escalated tickets are identified, but more non-escalated tickets are also flagged.

### R2.4 Logistic Regression vs KNN

At a threshold of 0.3, logistic regression achieves an F1 score of 0.5714 and recall of 0.7451.

The KNN model with K = 5 achieves an F1 score of 0.3696 and recall of 0.3333.

Therefore, logistic regression performs better than KNN on both F1 score and recall for this test set.

### R2.5 Cost-Based Model Selection

I would deploy the logistic regression model with an alert threshold of 0.15.

The evaluated test costs are:

- Selected threshold of 0.15: INR 6,050
- Threshold of 0.5: INR 11,800
- Flagging nobody: INR 20,400

The threshold was selected using training rows rather than test rows to prevent information leakage and preserve an honest evaluation on unseen data.

## Part 3 findings

### R3.1 K-means and Elbow Method

The successive WCSS drops are:

- WCSS(2) − WCSS(3) = 659.8282
- WCSS(3) − WCSS(4) = 108.0739
- WCSS(4) − WCSS(5) = 171.0575
- WCSS(5) − WCSS(6) = 45.5466

The elbow method supports K = 3 because the reduction from K = 3 to K = 4 is less than half the previous reduction. The drops do not shrink steadily because the pinned starting rows provide a reproducible solution, not necessarily the global optimum.

### R3.2 Segment Analysis

**Segment A – Stable Tickets:** Contains 322 records with an outcome rate of 1.86%. These tickets have higher customer tenure, lower SLA consumption and fewer earlier tickets. The support team should use normal resolution workflows and monitor these tickets routinely.

**Segment B – High-Value Repeat Tickets:** Contains 255 records with an outcome rate of 23.53%. These tickets have the highest mean account value and the highest earlier ticket count. The support team should prioritise repeat contacts and provide faster assistance to high-value accounts.

**Segment C – High-SLA-Risk Tickets:** Contains 223 records with an outcome rate of 46.64%, the highest among the three segments. These tickets have the highest SLA consumption and lower satisfaction scores. The support team should escalate these tickets for early review before SLA limits are reached.

### R3.3 Hierarchical Clustering

At the selected dendrogram cuts:

- Single linkage: E1, E2+E3+E6, E4, E5.
- Complete linkage: E1+E4+E5, E2+E3+E6.

Single linkage uses the minimum distance between cluster members, which allows nearby groups to connect through a chain of observations. Complete linkage uses the maximum distance, producing more compact clusters and keeping groups separate until their farthest members are sufficiently close.

### R3.4 Principal Component Analysis

The first principal component is mainly influenced by X1 (+0.5317), X3 (−0.5187), X5 (+0.4418) and X2 (−0.4100). This indicates that customer tenure and satisfaction contribute positively, while SLA consumption and earlier ticket count contribute negatively.

Three principal components are required to explain at least 80% of the variance, explaining 86.16% cumulatively.

The mean PC1 scores are:

- Segment A: +1.7773
- Segment B: −0.7831
- Segment C: −1.6709

This shows that Segment A has the highest PC1 score, while Segment C has the lowest. Segment B lies between the two.


## Part 4 findings

### R4.1 Decision Tree Split

The root entropy is 0.7462 and root Gini is 0.3347.

S4 (Customer tenure <= 24 months) gives the highest information gain of 0.1252 and Gini gain of 0.0606.

Therefore, S4 is the best split according to both criteria.

### R4.2 Model Comparison and Voting

Logistic regression achieves an F1 score of 0.5714 and recall of 0.7451 at threshold 0.3.

KNN achieves an F1 score of 0.3696 and recall of 0.3333.

The decision stump achieves the highest recall of 0.7647, while logistic regression gives the highest F1 score.

The hard voting model achieves an F1 score of 0.5344, while soft voting achieves 0.5507.

Therefore, logistic regression provides the best overall F1 performance among the evaluated models.

### R4.3 AdaBoost

The first-round weighted error is 0.2500 and alpha is 0.5493.

The second-round weighted error is 0.1667 and alpha is 0.8047.

The final ensemble accuracy is 75%.

The second weak learner receives greater importance because its weighted error is lower.

### R4.4 Out-of-Bag Share

The out-of-bag fraction is 0.3676, meaning approximately 36.76% of the training records are expected to remain unused in a bootstrap sample.

### R4.5 Neural Network Architecture

The MLP architecture requires 13,002 parameters, while the CNN requires 15,734 parameters.

For a 128 x 128 x 3 input, the fully connected layer requires 50,332,672 parameters, compared with only 1,792 parameters for the convolutional layer.

The RNN architecture for ticket classification requires 26,768 parameters.

## Model-choice memo

**To:** Head of Customer Support Operations  
**Subject:** Recommendation for Swiggy Support Decision Intelligence

I recommend deploying Logistic Regression as the primary structured-data model for predicting ticket escalations.

The model achieves an F1 score of 0.5714 and recall of 0.7451 at a threshold of 0.3, outperforming KNN, which achieves an F1 score of 0.3696 and recall of 0.3333. The decision stump achieves a slightly higher recall of 0.7647 but a lower F1 score of 0.5532.

For production, I recommend an alert threshold of 0.15, selected using training data to minimise operational costs. Its test cost is INR 6,050, compared with INR 11,800 at a threshold of 0.5 and INR 20,400 when no tickets are flagged.

The main limitation is that the dataset contains only 800 synthetic records, so performance must be validated on real Swiggy support data before deployment.

I recommend prioritising the structured-data model before developing Model A or Model B. Model B, the ticket-text classification system, would be a useful future extension once the structured model is established.

**Recommendation:** Deploy Logistic Regression with a 0.15 alert threshold after real-world validation.