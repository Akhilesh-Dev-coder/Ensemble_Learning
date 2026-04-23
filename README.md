# 🚢 Titanic Survival Predictor: Ensemble Learning Project

This project demonstrates the implementation of **Ensemble Learning Techniques** using the **Random Forest** algorithm to predict passenger survival on the Titanic. It includes a comprehensive theory document, a data analysis/modeling notebook, and a modern Streamlit front-end application.

## 🚀 Project Overview

The goal of this project is to build a robust classification model that can predict whether a passenger would have survived the Titanic disaster based on features like age, sex, passenger class, and more.

### Key Components:
- **Theory Documentation**: Detailed explanation of Bagging, Boosting, and Stacking.
- **Jupyter Notebook**: Complete data preprocessing, feature engineering, model training, and evaluation.
- **Streamlit Web App**: An interactive, user-friendly interface for real-time predictions.

## 🛠️ Technology Stack
- **Language**: Python 3.8+
- **Libraries**: 
  - `Scikit-learn` (Modeling)
  - `Pandas` & `NumPy` (Data Manipulation)
  - `Seaborn` & `Matplotlib` (Visualization)
  - `Streamlit` (Web Front-end)
  - `Joblib` (Model Serialization)

## 📊 Model Performance
The **Random Forest Classifier** was chosen for its ability to reduce variance and handle complex datasets.
- **Accuracy**: ~81% on the test set.
- **Features Used**: Passenger Class, Sex, Age, Siblings/Spouses Aboard, Parents/Children Aboard, Fare, and Port of Embarkation.

## 💻 Getting Started

### Prerequisites
Ensure you have Python installed. You will need to install the following dependencies:
```bash
pip install pandas seaborn scikit-learn matplotlib streamlit joblib
```

### Running the Project
1. **Train the Model**: Run the Jupyter Notebook `Ensemble_Learning_Techniques.ipynb` to generate the `random_forest_model.pkl` file.
2. **Launch the Web App**: Run the following command in your terminal:
```bash
streamlit run app.py
```

## 📁 Repository Structure
- `Ensemble_Learning_Techniques.ipynb`: The main modeling notebook.
- `app.py`: Streamlit application source code.
- `Ensemble_Learning_Theory.md`: Theoretical documentation on Ensemble Learning.
- `random_forest_model.pkl`: The serialized trained model.
- `README.md`: Project documentation.

## 📜 License
This project is open-source and available under the [MIT License](LICENSE).

---
*Developed as part of the Python AI Machine Learning curriculum.*
