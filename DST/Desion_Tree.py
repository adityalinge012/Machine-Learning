
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, StratifiedKFold

# ------------------------------------------------------------
# PART A: LOAD DATASET AND SELECT FEATURES
# ------------------------------------------------------------

g_iris = load_iris()

# Feature 2 = petal length
# Feature 3 = petal width
g_X = g_iris.data[:, [2, 3]]
g_y = g_iris.target

g_class_names = g_iris.target_names
g_feature_names = ["Petal length (cm)", "Petal width (cm)"]
g_n_classes = len(g_class_names)

print("=" * 60)
print("MANUAL DECISION TREE CLASSIFICATION")
print("=" * 60)
print("Total observations:", len(g_y))
print("Input features:", g_feature_names)
print("Classes:", list(g_class_names))

# ------------------------------------------------------------
# PART B: TRAIN/TEST SPLIT
# ------------------------------------------------------------

g_X_train, g_X_test, g_y_train, g_y_test = train_test_split(
    g_X,
    g_y,
    test_size=0.20,
    random_state=42,
    stratify=g_y
)

print("\nTRAIN/TEST SPLIT")
print("Training observations:", len(g_y_train))
print("Testing observations :", len(g_y_test))

# ------------------------------------------------------------
# PART C: MANUAL GINI IMPURITY AND BEST SPLIT
# ------------------------------------------------------------

def g_gini_impurity(g_labels):
    """Calculate Gini impurity for a group of class labels."""

    if len(g_labels) == 0:
        return 0.0

    g_counts = np.bincount(g_labels, minlength=g_n_classes)
    g_probabilities = g_counts / len(g_labels)

    return 1.0 - np.sum(g_probabilities ** 2)


def g_best_split(g_X_node, g_y_node):
    """
    Examine candidate thresholds for each feature.
    Return the split with the minimum weighted Gini impurity.
    """

    g_best_score = float("inf")
    g_best_feature = None
    g_best_threshold = None
    g_n_samples = len(g_y_node)

    if g_n_samples < 2:
        return None, None, None

    for g_feature in range(g_X_node.shape[1]):

        g_values = np.unique(g_X_node[:, g_feature])

        # Candidate thresholds are midpoints between
        # consecutive unique values.
        g_thresholds = (
            g_values[:-1] + g_values[1:]
        ) / 2.0

        for g_threshold in g_thresholds:

            g_left_mask = (
                g_X_node[:, g_feature] <= g_threshold
            )
            g_right_mask = ~g_left_mask

            if not np.any(g_left_mask) or not np.any(g_right_mask):
                continue

            g_y_left = g_y_node[g_left_mask]
            g_y_right = g_y_node[g_right_mask]

            g_left_weight = len(g_y_left) / g_n_samples
            g_right_weight = len(g_y_right) / g_n_samples

            g_split_score = (
                g_left_weight * g_gini_impurity(g_y_left)
                + g_right_weight * g_gini_impurity(g_y_right)
            )

            if g_split_score < g_best_score:
                g_best_score = g_split_score
                g_best_feature = g_feature
                g_best_threshold = float(g_threshold)

    return (
        g_best_feature,
        g_best_threshold,
        g_best_score
    )


# ------------------------------------------------------------
# PART D: MANUAL RECURSIVE DECISION TREE CONSTRUCTION
# ------------------------------------------------------------

def g_majority_class(g_labels):
    """Return the most frequent class in a node."""

    g_counts = np.bincount(
        g_labels,
        minlength=g_n_classes
    )

    return int(np.argmax(g_counts))


def g_build_tree(g_X_node, g_y_node, g_depth=0, g_max_depth=5):
    """
    Construct a binary decision tree recursively.

    Stop when:
    1. The node is pure.
    2. Maximum depth is reached.
    3. No valid split is available.
    4. The split does not reduce impurity.
    """

    if len(g_y_node) == 0:
        raise ValueError("Cannot build a tree from an empty node.")

    g_majority = g_majority_class(g_y_node)
    g_impurity = g_gini_impurity(g_y_node)

    # Stopping condition 1: pure node
    if g_impurity == 0.0:
        return {
            "leaf": True,
            "class": g_majority
        }

    # Stopping condition 2: maximum depth reached
    if g_depth >= g_max_depth:
        return {
            "leaf": True,
            "class": g_majority
        }

    (
        g_feature,
        g_threshold,
        g_split_score
    ) = g_best_split(g_X_node, g_y_node)

    # Stopping conditions 3 and 4
    if (
        g_feature is None
        or g_split_score is None
        or g_split_score >= g_impurity
    ):
        return {
            "leaf": True,
            "class": g_majority
        }

    g_left_mask = (
        g_X_node[:, g_feature] <= g_threshold
    )
    g_right_mask = ~g_left_mask

    # Store the decision and recursively construct children.
    return {
        "leaf": False,
        "feature": g_feature,
        "threshold": g_threshold,
        "left": g_build_tree(
            g_X_node[g_left_mask],
            g_y_node[g_left_mask],
            g_depth + 1,
            g_max_depth
        ),
        "right": g_build_tree(
            g_X_node[g_right_mask],
            g_y_node[g_right_mask],
            g_depth + 1,
            g_max_depth
        )
    }


# ------------------------------------------------------------
# PART E: MANUAL PREDICTION AND EVALUATION
# ------------------------------------------------------------

def g_predict_one(g_node, g_observation):
    """Predict the class of one observation."""

    if g_node["leaf"]:
        return g_node["class"]

    g_feature = g_node["feature"]
    g_threshold = g_node["threshold"]

    if g_observation[g_feature] <= g_threshold:
        return g_predict_one(g_node["left"], g_observation)

    return g_predict_one(g_node["right"], g_observation)


def g_predict(g_tree, g_X_input):
    """Predict classes for multiple observations."""

    return np.array([
        g_predict_one(g_tree, g_observation)
        for g_observation in g_X_input
    ], dtype=int)


def g_confusion_matrix(g_actual, g_predicted, g_n_classes=3):
    """
    Build a confusion matrix manually.
    Rows = actual classes.
    Columns = predicted classes.
    """

    g_matrix = np.zeros(
        (g_n_classes, g_n_classes),
        dtype=int
    )

    for g_actual_class, g_predicted_class in zip(
        g_actual, g_predicted
    ):
        g_matrix[g_actual_class, g_predicted_class] += 1

    return g_matrix


def g_evaluate(g_actual, g_predicted):
    """Calculate accuracy and macro-averaged metrics manually."""

    g_matrix = g_confusion_matrix(
        g_actual,
        g_predicted,
        g_n_classes=g_n_classes
    )

    g_total = np.sum(g_matrix)
    g_accuracy = (
        np.trace(g_matrix) / g_total
        if g_total > 0 else 0.0
    )

    g_precisions = []
    g_recalls = []
    g_f1_scores = []

    for g_class in range(g_n_classes):

        g_true_positive = g_matrix[g_class, g_class]

        g_false_positive = (
            np.sum(g_matrix[:, g_class]) - g_true_positive
        )

        g_false_negative = (
            np.sum(g_matrix[g_class, :]) - g_true_positive
        )

        g_precision_denominator = (
            g_true_positive + g_false_positive
        )
        g_recall_denominator = (
            g_true_positive + g_false_negative
        )

        g_precision = (
            g_true_positive / g_precision_denominator
            if g_precision_denominator > 0 else 0.0
        )

        g_recall = (
            g_true_positive / g_recall_denominator
            if g_recall_denominator > 0 else 0.0
        )

        g_f1_denominator = g_precision + g_recall

        g_f1 = (
            2 * g_precision * g_recall / g_f1_denominator
            if g_f1_denominator > 0 else 0.0
        )

        g_precisions.append(g_precision)
        g_recalls.append(g_recall)
        g_f1_scores.append(g_f1)

    g_results = {
        "accuracy": g_accuracy,
        "precision": np.mean(g_precisions),
        "recall": np.mean(g_recalls),
        "f1": np.mean(g_f1_scores)
    }

    return g_matrix, g_results


# Train the manually coded Decision Tree.
g_max_depth = 5

g_tree = g_build_tree(
    g_X_train,
    g_y_train,
    g_depth=0,
    g_max_depth=g_max_depth
)

# Predict the held-out test observations.
g_y_test_pred = g_predict(g_tree, g_X_test)

g_test_matrix, g_test_results = g_evaluate(
    g_y_test,
    g_y_test_pred
)

print("\nTEST SET CONFUSION MATRIX")
print("Rows = actual; columns = predicted")
print(g_test_matrix)

print("\nTEST SET PERFORMANCE")
for g_metric, g_value in g_test_results.items():
    print(f"{g_metric.capitalize():12s}: {g_value:.4f}")


# ------------------------------------------------------------
# PART F: STRATIFIED 5-FOLD CROSS-VALIDATION
# ------------------------------------------------------------

g_kfold = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

g_cv_results = {
    "accuracy": [],
    "precision": [],
    "recall": [],
    "f1": []
}

# Cross-validation uses training data only.
for g_fold, (g_train_idx, g_val_idx) in enumerate(
    g_kfold.split(g_X_train, g_y_train),
    start=1
):

    g_X_fold_train = g_X_train[g_train_idx]
    g_y_fold_train = g_y_train[g_train_idx]

    g_X_fold_val = g_X_train[g_val_idx]
    g_y_fold_val = g_y_train[g_val_idx]

    g_fold_tree = g_build_tree(
        g_X_fold_train,
        g_y_fold_train,
        g_depth=0,
        g_max_depth=g_max_depth
    )

    g_fold_pred = g_predict(
        g_fold_tree,
        g_X_fold_val
    )

    g_fold_matrix, g_fold_metrics = g_evaluate(
        g_y_fold_val,
        g_fold_pred
    )

    for g_metric in g_cv_results:
        g_cv_results[g_metric].append(
            g_fold_metrics[g_metric]
        )

    print(
        f"\nFold {g_fold}: "
        f"Accuracy = {g_fold_metrics['accuracy']:.4f}, "
        f"Precision = {g_fold_metrics['precision']:.4f}, "
        f"Recall = {g_fold_metrics['recall']:.4f}, "
        f"F1 = {g_fold_metrics['f1']:.4f}"
    )

print("\n5-FOLD CROSS-VALIDATION SUMMARY")

for g_metric, g_values in g_cv_results.items():
    print(
        f"{g_metric.capitalize():12s}: "
        f"{np.mean(g_values):.4f} "
        f"+/- {np.std(g_values):.4f}"
    )


# ------------------------------------------------------------
# PART G: VISUALIZE DECISION REGIONS AND PERFORMANCE
# ------------------------------------------------------------

# Construct a grid covering the feature space.
g_x_min = g_X[:, 0].min() - 0.5
g_x_max = g_X[:, 0].max() + 0.5
g_y_min = g_X[:, 1].min() - 0.3
g_y_max = g_X[:, 1].max() + 0.3

g_x_values = np.linspace(g_x_min, g_x_max, 300)
g_y_values = np.linspace(g_y_min, g_y_max, 300)

g_xx, g_yy = np.meshgrid(
    g_x_values,
    g_y_values
)

g_grid = np.c_[
    g_xx.ravel(),
    g_yy.ravel()
]

# Predict a class for every point on the grid.
g_grid_predictions = g_predict(
    g_tree,
    g_grid
).reshape(g_xx.shape)

# Create a two-panel visualization.
g_fig, g_axes = plt.subplots(
    1, 2,
    figsize=(14, 5.5)
)

# Panel 1: Decision regions and observations.
g_axes[0].contourf(
    g_xx,
    g_yy,
    g_grid_predictions,
    levels=[-0.5, 0.5, 1.5, 2.5],
    alpha=0.30
)

for g_class in range(g_n_classes):

    g_class_mask = g_y == g_class

    g_axes[0].scatter(
        g_X[g_class_mask, 0],
        g_X[g_class_mask, 1],
        label=g_class_names[g_class],
        edgecolors="black",
        s=35
    )

g_axes[0].set_xlabel(g_feature_names[0])
g_axes[0].set_ylabel(g_feature_names[1])
g_axes[0].set_title(
    "Manual Decision Tree: Decision Regions"
)
g_axes[0].legend()
g_axes[0].grid(alpha=0.2)

# Panel 2: Test metrics and cross-validation means.
g_metric_names = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-score"
]

g_metric_keys = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

g_test_values = [
    g_test_results[g_key]
    for g_key in g_metric_keys
]

g_cv_means = [
    np.mean(g_cv_results[g_key])
    for g_key in g_metric_keys
]

g_positions = np.arange(len(g_metric_names))
g_bar_width = 0.36

g_axes[1].bar(
    g_positions - g_bar_width / 2,
    g_test_values,
    g_bar_width,
    label="Test set"
)

g_axes[1].bar(
    g_positions + g_bar_width / 2,
    g_cv_means,
    g_bar_width,
    label="5-fold CV mean"
)

g_axes[1].set_xticks(g_positions)
g_axes[1].set_xticklabels(g_metric_names)
g_axes[1].set_ylim(0, 1.10)
g_axes[1].set_ylabel("Score")
g_axes[1].set_title("Classification Performance")
g_axes[1].legend()
g_axes[1].grid(axis="y", alpha=0.2)

# Label each bar with its score.
for g_container in g_axes[1].containers:
    g_axes[1].bar_label(
        g_container,
        fmt="%.3f",
        padding=2,
        fontsize=8
    )

g_fig.suptitle(
    "Iris Classification Using a Manually Coded Decision Tree",
    fontsize=13
)

g_fig.tight_layout()

g_output_file = "iris_decision_tree_visualization.png"

g_fig.savefig(
    g_output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nVisualization saved as:", g_output_file)
print("\nExperiment completed successfully.")
