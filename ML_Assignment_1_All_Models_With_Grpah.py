import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score, confusion_matrix

# Load the dataset
data = pd.read_csv('C:\\Python\\ML_Assignment\\dataset\\heart.csv')

# Fill missing values with the mean of each column if there are any
data.fillna(data.mean(), inplace=True)

# Splitting the dataset into features (X) and target variable (y)
X = data.iloc[:, :-1]  # All columns except the last one as features
y = data.iloc[:, -1]   # The last column as target

# Split the data into training (70%) and temporary set (30%) for further splitting
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)

# Split the temporary set into validation (15%) and testing (15%) sets
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scaling the features for better performance of the models
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # Fit and transform the training set
X_val = scaler.transform(X_val)          # Only transform the validation set
X_test = scaler.transform(X_test)        # Only transform the testing set

# Initialize various models
models = {
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Naïve Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(random_state=42)
}

# Dictionary to store results for each model
results = {}

# Train, predict, and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)  # Train the model
    predictions = model.predict(X_val)  # Predict on the validation set
    # Store evaluation metrics
    results[name] = {
        'Accuracy': accuracy_score(y_val, predictions),
        'Precision': precision_score(y_val, predictions, zero_division=0),
        'Recall': recall_score(y_val, predictions, zero_division=0),
        'F1 Score': f1_score(y_val, predictions, zero_division=0)
    }

# Implementing Bagging with Decision Tree as the base estimator
bagging_model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), n_estimators=50, random_state=42)
bagging_model.fit(X_train, y_train)  # Train the bagging model
bagging_predictions = bagging_model.predict(X_val)  # Predict on the validation set

# Evaluate the Bagging model
bagging_results = {
    'Accuracy': accuracy_score(y_val, bagging_predictions),
    'Precision': precision_score(y_val, bagging_predictions, zero_division=0),
    'Recall': recall_score(y_val, bagging_predictions, zero_division=0),
    'F1 Score': f1_score(y_val, bagging_predictions, zero_division=0)
}

# Adding Bagging results to the results dictionary
results["Bagging with Decision Tree"] = bagging_results

# Display the results for each model
for name, metrics in results.items():
    print(f"\n{name}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

# Plotting the ROC curves for classifiers with predict_proba method
plt.figure(figsize=(10, 8))

classifiers = [
    ('Decision Tree', models["Decision Tree"]),
    ('Random Forest', models["Random Forest"]),
    ('Naive Bayes', models["Naïve Bayes"]),
    ('KNN', models["K-Nearest Neighbors"]),
    ('Logistic Regression', models["Logistic Regression"]),
    ('Bagging', bagging_model)
]

for name, clf in classifiers:
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')

plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

# Plotting the evaluation metrics in a graph paper style
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
fig, ax = plt.subplots(2, 2, figsize=(14, 10))

# Create a plot for each metric
for i, metric in enumerate(metrics_names):
    values = [results[name][metric] for name in results.keys()]
    ax[i//2, i%2].bar(results.keys(), values)
    ax[i//2, i%2].set_title(metric)
    ax[i//2, i%2].set_ylim(0, 1)
    ax[i//2, i%2].set_ylabel(metric)
    ax[i//2, i%2].set_xticklabels(results.keys(), rotation=45, ha='right')
    ax[i//2, i%2].grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
    ax[i//2, i%2].set_facecolor('#f0f0f0')  # Set background color to mimic graph paper

plt.tight_layout()
plt.show()

# Plotting the confusion matrix for each classifier
for name, clf in classifiers:
    predictions = clf.predict(X_test)
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()
