import matplotlib.pyplot as plt
import seaborn as sns
import os   

def plot_class_distribution(data, target_column, output_dir):
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x=target_column)
    plt.title(f'Distribution of {target_column}')
    plt.xlabel(target_column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{target_column}_distribution.png'))
    plt.show()

def plot_numerical_distribution(data, numerical_columns, output_dir):
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.histplot(data[column], kde=True, bins=30)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{column}_distribution.png'))
        plt.show()  

def plot_correlation_heatmap(data, output_dir):
    plt.figure(figsize=(12, 10))
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.show()

def plot_boxplot(data, numerical_columns, output_dir):
    for column in numerical_columns:
        plt.figure(figsize=(8, 6))
        sns.boxplot(data=data, x=column)
        plt.title(f'Boxplot of {column}')
        plt.xlabel(column)
        plt.tight_layout()
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{column}_boxplot.png'))
        plt.show()

def plot_target_vs_age(data, target_column, age_column, output_dir):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=data, x=target_column, y=age_column)
    plt.title(f'{target_column} vs {age_column}')
    plt.xlabel(target_column)
    plt.ylabel(age_column)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f'{target_column}_vs_{age_column}.png'))
    plt.show()          


def eda_pipeline(data, target_column, numerical_columns, age_column, output_dir):
    plot_class_distribution(data, target_column, output_dir)
    plot_numerical_distribution(data, numerical_columns, output_dir)
    plot_correlation_heatmap(data, output_dir)
    plot_boxplot(data, numerical_columns, output_dir)
    plot_target_vs_age(data, target_column, age_column, output_dir)
    print("EDA completed successfully.")
    