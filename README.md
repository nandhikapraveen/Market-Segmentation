# Market Segmentation Analysis

This project focuses on understanding customer behavior and segmenting the customer base using various clustering techniques. The dataset used in this project contains information about customer transactions and behaviors.

## Getting Started

### Prerequisites

Ensure you have the following libraries installed:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages via pip:

    pip install pandas numpy matplotlib seaborn scikit-learn

### Installation

Clone the repository to your local machine:

    git clone https://github.com/nandhikapraveen/Market-Segmentation
    cd your-repository-directory

### Usage

Run the Jupyter notebook to see the analysis:

    jupyter notebook Market_Segmentation_Customer_Data.ipynb

## Analysis Workflow

1. **Data Loading**: Load the customer data from a CSV file.
2. **Exploratory Data Analysis (EDA)**: Analyze the data to find patterns, relationships, or anomalies to address.
3. **Data Preprocessing**:
   - Scale the data using `StandardScaler`.
   - Handle missing values by imputing the mean.
   - Drop unnecessary columns.
4. **Dimensionality Reduction**:
   - Apply PCA to reduce dimensions for better visualization and clustering performance.
5. **Clustering**:
   - Use KMeans and other clustering algorithms like Agglomerative Clustering and DBSCAN.
   - Determine the optimal number of clusters using the Elbow method.
   - Visualize the clustering to understand customer segments.
6. **Model Evaluation**:
   - Evaluate clustering performance using silhouette scores.
7. **Model Persistence**:
   - Save the trained models using joblib for future prediction tasks.

## Visualization

Visualizations are provided for:
- Distribution of various features.
- Correlation heatmap of the features.
- PCA results and clustering visualizations.

## Saving the Model

Models are saved to disk using the `joblib` library, allowing for model deployment or further evaluation later.

## Authors

- **Nandhika Praveen**

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Thanks to everyone who has contributed to this project.
