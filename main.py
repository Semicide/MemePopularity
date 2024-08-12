import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
# Load the dataset from the Excel file
df = pd.read_excel("ml3_train.xlsx")

# Define labels for each feature
humor_type_labels = ['Absurdism', 'Dark', 'Sarcasm/Irony', 'Rage-Bait', 'Wholesome', 'Satirical', 'Dank', 'Relatable']
format_labels = ['Photo', 'Video', 'Text']
featuring_song_labels = ['Yes', 'No']
visual_features_labels = ['Multiple_Features', 'Reaction', 'Absurd', 'Animals', 'Pop_Culture', 'Surreal', 'Deep_Fried']
op_platform_labels = ['Twitter(X)', 'Instagram', 'TikTok', 'Unknown', 'Youtube', 'FaceBook', 'Reddit']


# Map the labels to the corresponding features
label_mapping = {
    'Humor_Type': {label: idx for idx, label in enumerate(humor_type_labels)},
    'Format': {label: idx for idx, label in enumerate(format_labels)},
    'Song': {label: idx for idx, label in enumerate(featuring_song_labels)},
    'Visual_Features': {label: idx for idx, label in enumerate(visual_features_labels)},
    'OP_Platform': {label: idx for idx, label in enumerate(op_platform_labels)},
    'Is_Popular': {'No': 0, 'Yes': 1}  # Assuming 'No' is 0 and 'Yes' is 1
}

# Apply label encoding
for column, mapping in label_mapping.items():
    df[column] = df[column].map(mapping)

# Features and target variable
X = df.drop(['Is_Popular', 'Op_Link or related link', 'Description'], axis=1)
y = df['Is_Popular']

# Decision Tree model with entropy
model = DecisionTreeClassifier(criterion='entropy')
accuracy_scores = []
# K-fold cross-validation
kf = KFold(n_splits=4, shuffle=True, random_state=42)
# Perform k-fold cross-validation
for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y), 1):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test)

    # Calculate accuracy for the current fold
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_scores.append(accuracy)

    # Print descriptions and predictions for each entry in the test set
    print(f"\nDescriptions and Predictions - Fold {fold_idx}:")
    for index, (description, prediction) in enumerate(zip(df['Description'].iloc[test_index], y_pred), 1):
        print(f"Entry {index} - Description: {description}, Prediction: {prediction}")

    # Calculate and print confusion matrix for each fold
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix - Fold {fold_idx}:")
    print(conf_matrix)

    # Print predictions for each fold
    print(f"\nPredictions - Fold {fold_idx}:")
    print(y_pred)

    # Print classification report for each fold
    classification_rep = classification_report(y_test, y_pred)
    print(f"\nClassification Report - Fold {fold_idx}:\n", classification_rep)

# Predict on the entire dataset to get predictions for the decision tree
y_pred_all = model.predict(X)

# Print predictions for the entire dataset
print("\nPredictions - Entire Dataset:")
print(y_pred_all)

# Print corresponding entries for the entire dataset
print("\nCorresponding Entries - Entire Dataset:")
print(df.iloc[:, :7])  # Assuming the first three columns are the entries you want to display

# Calculate and print the average accuracy
avg_accuracy = sum(accuracy_scores) / len(accuracy_scores)
print(f"\nAverage Accuracy Across Folds: {avg_accuracy:.4f}")
# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, class_names=['Not Popular', 'Popular'], filled=True, rounded=True)
plt.show()
