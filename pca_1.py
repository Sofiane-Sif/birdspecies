import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Read the CSV file
data = pd.read_csv('data.csv')

def f1():
    # Separate the months and bird data
    mois = data.iloc[:, 0]  # Column containing the months
    donnees = data.iloc[:, 1:]  # Columns containing the bird data

    # Standardize the data
    scaler = StandardScaler()
    donnees_std = scaler.fit_transform(donnees)

    # Perform PCA on the standardized data
    pca = PCA(n_components=2)
    composantes_principales = pca.fit_transform(donnees_std)

    # Create a figure
    fig, ax = plt.subplots()

    # Plot the principal components with different colors for each month
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow', 'lime', 'pink', 'brown', 'gray']
    labels = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

    for i, label in enumerate(labels):
        ax.scatter(composantes_principales[i, 0], composantes_principales[i, 1], c=colors[i], label=label)
        ax.annotate(label, (composantes_principales[i, 0], composantes_principales[i, 1]))


    # Add the percentage of variance explained by each component on the axes
    variance_exp = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({variance_exp[0]*100:.2f}%)')
    ax.set_ylabel(f'PC2 ({variance_exp[1]*100:.2f}%)')

    # Show the figure
    plt.show()

def f2():
    # Separate the months and bird data
    mois = data.iloc[:, 0]  # Column containing the months
    donnees = data.iloc[:, 1:]  # Columns containing the bird data

    # Perform PCA
    pca = PCA()
    pca.fit(donnees)

    variances = pca.explained_variance_ratio_ * 100

    # Plot the variances
    plt.bar(range(1, len(variances) + 1), variances)
    plt.xlabel('Variable')
    plt.ylabel('Variance (%)')
    plt.title('Variance of Each Variable in PCA Space')
    plt.show()
f1()
f2()
