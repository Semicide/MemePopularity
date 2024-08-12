import pandas as pd
import math
import random
import sys


class TreeNode:
    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.children = {}
        self.feature = None
        self.threshold = None
        self.label = None


def decision_tree_train(data, target, node_counter, max_depth, current_depth=0):
    node = TreeNode(data, target)

    if len(set(target)) == 1:
        node.label = target.iloc[0]
        return node

    if len(data.columns) == 0 or current_depth == max_depth:
        node.label = target.mode().iloc[0]
        return node

    best_feature, best_threshold = find_best_split(data, target)

    branches = split_data(data, target, best_feature, best_threshold)

    node.feature = best_feature
    node.threshold = best_threshold

    for value, branch_data in branches.items():
        node.children[value] = decision_tree_train(
            branch_data['data'], branch_data['target'], node_counter, max_depth, current_depth + 1
        )

    if not any(child.label is None for child in node.children.values()):
        node_counter[0] += 1  # Increment the node counter only if there are children

    return node




def find_best_split(data, target):
    best_feature = None
    best_threshold = None
    best_entropy = float('inf')

    for feature in data.columns:
        thresholds = data[feature].unique()

        random.shuffle(thresholds)

        for threshold in thresholds:
            branches = split_data(data, target, feature, threshold)
            entropy = calculate_entropy(branches)

            if entropy == best_entropy:
                if random.choice([True, False]):
                    best_feature = feature
                    best_threshold = threshold
            elif entropy < best_entropy:
                best_entropy = entropy
                best_feature = feature
                best_threshold = threshold

    print(f"Entropy for {best_feature} at threshold {best_threshold}: {best_entropy}")

    return best_feature, best_threshold


def split_data(data, target, feature, threshold):
    branches = {}
    unique_values = data[feature].dropna().unique()  # Exclude None values
    for value in unique_values:
        mask = data[feature] == value
        branches[value] = {
            'data': data[mask],
            'target': target[mask]
        }
    return branches


def calculate_entropy(branches):
    total_samples = sum(len(branch['target']) for branch in branches.values())
    entropy = sum((len(branch['target']) / total_samples) * calculate_single_entropy(branch['target']) for branch in branches.values())
    return entropy


def calculate_single_entropy(target):
    class_counts = target.value_counts()
    total_samples = len(target)
    entropy = -sum((count / total_samples) * log2(count / total_samples) for count in class_counts)
    return entropy


def log2(x):
    return 0 if x == 0 else math.log2(x)


def classify_instance(instance, node):
    if node.label is not None:
        return node.label
    elif instance[node.feature] in node.children:
        return classify_instance(instance, node.children[instance[node.feature]])
    else:
        return None


def print_tree(node, indent=""):
    if node.label is not None:
        print(f"{indent}Is_Popular => {node.label}")
    else:
        print(f"{indent}Split on {node.feature} <= {node.threshold}")
        for value, child_node in node.children.items():
            print(f"{indent}  Value {value}:")
            print_tree(child_node, indent + "    ")




# Save tree structure to a text file
def save_tree_to_file(node, filename, predicted_column=None):
    with open(filename, 'w') as file:
        write_tree_to_file(node, file, predicted_column)


def write_tree_to_file(node, file, indent="", predicted_column=None):
    if node.label is not None:
        file.write(f"{indent}Is_Popular => {node.label}\n")
    else:
        file.write(f"{indent}Split on {node.feature} <= {node.threshold}\n")
        for value, child_node in node.children.items():
            file.write(f"{indent}  Value {value}:\n")
            write_tree_to_file(child_node, file, indent + "    ", predicted_column)


# Load the training and test data
train_data = pd.read_excel("ml3_train_labeled.xlsx")
# Initialize the node counter
node_counter = [0]
def count_nodes(node):
    if node is None:
        return 0

    count = 1  # Count the current node
    stack = list(node.children.values())  # Start with the children of the root

    while stack:
        current_node = stack.pop()
        count += 1

        # Add the children of the current node to the stack
        stack.extend(list(current_node.children.values()))

    return count


# Set the number of folds (e.g., 5)
k_folds = 5

# Shuffle the dataset
train_data = train_data.sample(frac=1, random_state=42).reset_index(drop=True)

# Split the data into k folds
fold_size = len(train_data) // k_folds
folds = [train_data[i * fold_size:(i + 1) * fold_size] for i in range(k_folds)]

# Initialize the node counter
node_counter = [0]

# Perform k-fold cross-validation
for i in range(k_folds):
    # Split the data into training and test sets
    test_data = folds[i]
    train_data = pd.concat([fold for j, fold in enumerate(folds) if j != i])

    # Train the decision tree on the training data
    target = train_data["Is_Popular"]
    features = train_data.drop("Is_Popular", axis=1)
    tree = decision_tree_train(features, target, node_counter, max_depth=10)

    # Print the tree structure for each fold
    print(f"\nTree structure for Fold {i + 1}:")
    print_tree(tree)

    # Print the number of nodes for each fold
    print(f"Number of nodes in the decision tree for Fold {i + 1}: {count_nodes(tree)}")

    # Classify the test instances for each fold
    test_predictions = []
    for _, row in test_data.iterrows():
        prediction = classify_instance(row, tree)
        test_predictions.append(prediction)

    # Add predictions to the test data
    test_data["Predicted_Popularity"] = test_predictions  # Change column name here

    # Print the classification results for each fold
    print(f"\nClassification results for Fold {i + 1}:")
    for index, row in test_data.iterrows():
        prediction = classify_instance(row, tree)
        print(f"Instance {index + 1}: Predicted Acceptability - {prediction}")

    # Calculate accuracy for each fold
    correct_predictions = (test_data["Is_Popular"] == test_data["Predicted_Popularity"]).sum()
    total_instances = len(test_data)
    accuracy = correct_predictions / total_instances
    print(f"Accuracy for Fold {i + 1}: {accuracy * 100:.2f}%")

    # Save the tree structure to a file for each fold
    save_tree_to_file(tree, f"decision_tree_fold_{i + 1}.txt", predicted_column="Predicted_Popularity")

# Save the results to an Excel file (optional)
# test_data.to_excel("classification_results.xlsx", index=False)