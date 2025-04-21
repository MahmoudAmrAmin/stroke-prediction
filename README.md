# Stroke Risk Prediction using Machine Learning

Stroke is a leading cause of death and disability worldwide. Early detection and prevention can significantly reduce its impact. This project leverages machine learning techniques to predict stroke risk in patients based on healthcare data, aiming to assist healthcare professionals in identifying high-risk individuals for timely intervention.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To set up this project locally, ensure you have Python 3.x installed along with the following dependencies:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

Install the required libraries using:

```bash
pip install -r requirements.txt
```

Clone the repository:

```bash
git clone https://github.com/MahmoudAmrAmin/stroke-prediction.git
cd stroke-prediction
```

## Usage

1. **Prepare the Environment**: Ensure all dependencies are installed as described above.
2. **Run the Notebooks**:
   - Open `preprocessing.ipynb` in Jupyter Notebook to preprocess the dataset:
     ```bash
     jupyter notebook preprocessing.ipynb
     ```
     Execute the cells sequentially to clean and prepare the data.
   - Open `models.ipynb` to train and evaluate machine learning models:
     ```bash
     jupyter notebook models.ipynb
     ```
     Run the cells to reproduce the modeling and evaluation steps.

## Project Structure

- `preprocessing.ipynb`: Notebook for data cleaning, feature engineering, and preprocessing.
- `models.ipynb`: Notebook for training, evaluating, and comparing machine learning models.
- `dataset/healthcare-dataset-stroke-data.csv`: Original dataset (not included in repo; source separately).
- `dataset/cleaned.csv`: Preprocessed dataset output from `preprocessing.ipynb`.
- `requirements.txt`: List of Python dependencies.
- `README.md`: Project documentation (this file).
- `LICENSE`: MIT License file.

## Data

The project utilizes the **Healthcare Dataset for Stroke Prediction**, sourced from [Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset). It contains 5,110 records with features such as:

- **Demographics**: Age, gender, marital status, residence type.
- **Medical History**: Hypertension, heart disease, average glucose level, BMI.
- **Lifestyle Factors**: Work type, smoking status.
- **Target**: Stroke occurrence (binary: 0 = no stroke, 1 = stroke).

### Preprocessing Steps
In `preprocessing.ipynb`, the data undergoes:
- Removal of irrelevant columns (e.g., `id`).
- Handling missing values (e.g., imputing BMI).
- Encoding categorical variables (e.g., one-hot encoding for gender, smoking status).
- Scaling numerical features (e.g., age, glucose level, BMI) for model compatibility.
- Visualization of biometric features (e.g., box plots for glucose levels and BMI by stroke status).

The cleaned dataset is saved as `dataset/cleaned.csv`.

## Models

In `models.ipynb`, the following machine learning models are implemented and evaluated:

- **Logistic Regression**
- **Naive Bayes (Gaussian)**
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**

### Evaluation Metrics
Models are assessed using:
- Accuracy
- Precision
- Recall
- F1-Score

The dataset is split into 80% training and 20% testing sets, with stratification to address class imbalance.

## Results

The **Logistic Regression** model achieved the highest accuracy at **95.2%**. However, due to the dataset's imbalance (stroke cases are rare), it exhibits a low recall for the positive class (stroke), as shown below:

| Model              | Accuracy | Precision | Recall | F1-Score |
|--------------------|----------|-----------|--------|----------|
| Logistic Regression| 0.952    | 1.000     | 0.02   | 0.039    |
| K-Nearest Neighbors| 0.951    | 0.500     | 0.04   | 0.074    |
| SVM                | 0.951    | 0.000     | 0.00   | 0.000    |
| Random Forest      | 0.948    | 0.000     | 0.00   | 0.000    |
| Decision Tree      | 0.908    | 0.133     | 0.16   | 0.145    |
| Naive Bayes        | 0.191    | 0.056     | 0.98   | 0.106    |

### Key Insights
- **Class Imbalance**: High accuracy masks poor performance on the minority class (stroke = 1).
- **Best Model**: Logistic Regression excels in accuracy but requires improvement in recall.
- Visualizations (e.g., confusion matrices) are available in `models.ipynb`.

## Future Work

- **Address Imbalance**: Implement techniques like SMOTE, class weighting, or oversampling.
- **Advanced Models**: Explore ensemble methods (e.g., XGBoost) or deep learning approaches.
- **Feature Engineering**: Incorporate additional features or external datasets for enhanced prediction.
- **Hyperparameter Tuning**: Use GridSearchCV to optimize model parameters.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a pull request.

Please report issues or suggestions via the [Issues](https://github.com/MahmoudAmrAmin/stroke-prediction/issues) page.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset provided by [fedesoriano](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) on Kaggle.
- Thanks to the open-source community for tools like `scikit-learn`, `pandas`, and `matplotlib`.
```

---

