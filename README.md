
# Level 2 NCEA Endorsement Classifier

This project aims to predict NCEA Level 2 endorsements (Achieved, Merit, Excellence) for students based on their internal project work using a Support Vector Machine (SVM) classifier. The data is sourced from the KAMAR school management system.

## Table of Contents
- [Installation](#installation)
- [Data Import](#data-import)
- [Data Cleanup](#data-cleanup)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Anaconda
- Python 3.8+

### Steps
1. Install [Anaconda](https://www.anaconda.com/products/individual) following the instructions for your operating system.

2. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/NCEA_Level2_Endorsement_Classifier.git
    cd NCEA_Level2_Endorsement_Classifier
    ```

3. Create and activate a new Anaconda environment:
    ```bash
    conda create -n ncea_env python=3.8
    conda activate ncea_env
    ```

4. Install the required libraries:
    ```bash
    pip install -r requirements.txt
    ```

## Data Import

### Exporting CSV Files from KAMAR
1. Use the standard export features in KAMAR to export the following CSV files:
    - Level2_2019.csv
    - Level2_2020.csv
    - Level2_2021.csv
    - Level2_2022.csv
    - Level2_2023.csv
    - Effort2019.csv
    - Effort2020.csv
    - Effort2021.csv
    - Effort2022.csv
    - Effort2023.csv

### Loading the Data
The data is loaded from CSV files and the columns are extracted as follows:

```python
import pandas as pd

# Load the data from CSV files
data_2019 = pd.read_csv('Level2_2019.csv')
data_2020 = pd.read_csv('Level2_2020.csv')
data_2021 = pd.read_csv('Level2_2021.csv')
data_2022 = pd.read_csv('Level2_2022.csv')
data_2023 = pd.read_csv('Level2_2023.csv')

# Load the effort data from CSV files
effort_2019 = pd.read_csv('Effort2019.csv')
effort_2020 = pd.read_csv('Effort2020.csv')
effort_2021 = pd.read_csv('Effort2021.csv')
effort_2022 = pd.read_csv('Effort2022.csv')
effort_2023 = pd.read_csv('Effort2023.csv')

# Ensure consistent column naming for 'Student ID'
data_2020.rename(columns={'Student ID': 'StudentID'}, inplace=True)
data_2021.rename(columns={'Student ID': 'StudentID'}, inplace=True)
data_2022.rename(columns={'Student ID': 'StudentID'}, inplace=True)
data_2023.rename(columns={'Student ID': 'StudentID'}, inplace=True)
```

## Data Cleanup

### Handling Missing Values
The data is cleaned by checking for and removing rows with missing values or zero values in key columns:

```python
# Check for missing values
missing_values = combined_data.isnull().sum()
print("Missing values in each column:\n", missing_values)

# Remove rows with any missing values
combined_data.dropna(inplace=True)

# Remove rows where all specified columns have zero values
columns_to_check = ['Level 2 Credits Not Achieved', 'Level 2 Credits Achieved', 
                    'Level 2 Credits Merit', 'Level 2 Credits Excellence']

combined_data = combined_data[~(combined_data[columns_to_check] == 0).all(axis=1)]

# Verify that there are no missing values left
missing_values_after = combined_data.isnull().sum()
print("\nMissing values after removing:\n", missing_values_after)
```

## Feature Engineering

### Calculating Endorsements
Apply the required logic for calculating Level 2 NCEA endorsements for students:

```python
def calculate_endorsement(row):
    if row['Level 2 Credits Excellence'] >= 50:
        return 'Excellence'
    elif row['Level 2 Credits Merit'] + row['Level 2 Credits Excellence'] >= 50:
        return 'Merit'
    elif row['Level 2 Credits Achieved'] + row['Level 2 Credits Merit'] + row['Level 2 Credits Excellence'] >= 50:
        return 'Achieved'
    else:
        return 'Not Achieved'

combined_data['Endorsement'] = combined_data.apply(calculate_endorsement, axis=1)
```

## Model Training

### Training the SVM Classifier
The SVM classifier is trained on the cleaned and feature-engineered dataset:

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# Split the data into training and test sets
X = combined_data.drop('Endorsement', axis=1)
y = combined_data['Endorsement']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Make predictions
y_pred = svm_classifier.predict(X_test)
```

## Evaluation

### Model Performance
Evaluate the model performance using a classification report and confusion matrix:

```python
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

## Usage

### Running the Jupyter Notebook
To run the notebook and reproduce the results, execute the following command in the project directory:

```bash
jupyter notebook AMEClassifier_S02E01.ipynb
```

Follow the instructions in the notebook to load data, clean data, train the model, and evaluate its performance.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
