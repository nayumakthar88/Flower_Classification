import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# --------------------------------------------------
# 1. Project Paths
# --------------------------------------------------

current_folder = os.path.dirname(os.path.abspath(__file__))
project_folder = os.path.dirname(current_folder)

data_folder = os.path.join(project_folder, "data")
output_folder = os.path.join(project_folder, "outputs")

csv_file = os.path.join(data_folder, "Iris.csv")

os.makedirs(output_folder, exist_ok=True)


# --------------------------------------------------
# 2. Load Dataset
# --------------------------------------------------

df = pd.read_csv(csv_file)

print("First 5 Rows\n")
print(df.head())

print("\nDataset Information\n")
df.info()

print("\nMissing Values\n")
print(df.isnull().sum())


# --------------------------------------------------
# 3. Encode Target Variable
# --------------------------------------------------

encoder = LabelEncoder()
df["species"] = encoder.fit_transform(df["species"])


# --------------------------------------------------
# 4. Split Features and Target
# --------------------------------------------------

X = df.drop("species", axis=1)
y = df["species"]


# --------------------------------------------------
# 5. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# --------------------------------------------------
# 6. Train Random Forest Model
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------

y_pred = model.predict(X_test)


# --------------------------------------------------
# 8. Model Evaluation
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))


# --------------------------------------------------
# 9. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "confusion_matrix.png")
)

plt.show()
plt.close()


# --------------------------------------------------
# 10. Species Count
# --------------------------------------------------

df_plot = pd.read_csv(csv_file)

plt.figure(figsize=(6, 4))

sns.countplot(
    x="species",
    data=df_plot
)

plt.title("Species Count")
plt.xlabel("Species")
plt.ylabel("Count")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "species_count.png")
)

plt.show()
plt.close()


# --------------------------------------------------
# 11. Pairplot
# --------------------------------------------------

pairplot = sns.pairplot(
    df_plot,
    hue="species"
)

pairplot.fig.suptitle(
    "Iris Species Pairplot",
    y=1.02
)

pairplot.savefig(
    os.path.join(output_folder, "species_pairplot.png")
)

plt.show()
plt.close()


# --------------------------------------------------
# 12. Correlation Heatmap
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.heatmap(
    df.drop("species", axis=1).corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(output_folder, "correlation_heatmap.png")
)

plt.show()
plt.close()


# --------------------------------------------------
# 13. Feature Distributions
# --------------------------------------------------

features = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width"
]

for feature in features:

    plt.figure(figsize=(6, 4))

    sns.histplot(
        df_plot[feature],
        bins=20,
        kde=True
    )

    plt.title(feature.replace("_", " ").title())
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            output_folder,
            f"{feature}.png"
        )
    )

    plt.show()
    plt.close()


# --------------------------------------------------
# 14. Completion Message
# --------------------------------------------------

print("\nProject Completed Successfully!")