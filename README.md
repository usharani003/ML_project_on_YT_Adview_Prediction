# YouTube Adview Prediction

## 📌 Project Overview

This project was completed during my **Machine Learning Internship at Internship Studio** from **27 April 2024 to 1 June 2024**.

The goal of this project was to build a machine learning model that predicts the number of **YouTube ad views (adviews)** based on various video metrics such as views, likes, dislikes, comments, duration, publishing date, and category.

Through this project, I gained hands-on experience in data preprocessing, exploratory data analysis (EDA), feature engineering, regression modeling, and model evaluation.

---

## 🎯 Objective

To develop a machine learning regression model capable of predicting YouTube adview counts using other available YouTube video metrics.

---

## 📊 Dataset Information

The dataset contains information about approximately **15,000 YouTube videos**.

### Features

| Feature   | Description                          |
| --------- | ------------------------------------ |
| vidid     | Unique ID of the video               |
| views     | Number of video views                |
| likes     | Number of likes                      |
| dislikes  | Number of dislikes                   |
| comment   | Number of comments                   |
| published | Video upload date                    |
| duration  | Video duration                       |
| category  | Video category                       |
| adview    | Number of ad views (Target Variable) |

### Problem Statement

YouTube advertisers pay content creators based on ad views and clicks. Predicting ad views can help estimate advertisement performance and content reach.

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* TensorFlow / Keras
* Jupyter Notebook

---

## 🔄 Project Workflow

### 1. Data Collection

* Loaded and explored the dataset
* Checked data types and dataset dimensions

### 2. Exploratory Data Analysis (EDA)

* Generated visualizations
* Studied feature distributions
* Analyzed correlations using heatmaps

### 3. Data Cleaning

* Removed missing values
* Handled invalid records
* Prepared data for modeling

### 4. Data Transformation

* Converted categorical features into numerical format
* Processed duration and date-related features

### 5. Data Preparation

* Feature scaling and normalization
* Train-validation-test split

### 6. Model Building

Implemented and compared multiple regression algorithms:

* Linear Regression
* Support Vector Regressor (SVR)
* Decision Tree Regressor
* Random Forest Regressor
* Artificial Neural Network (ANN)

### 7. Model Evaluation

Models were evaluated using regression error metrics and prediction performance.

### 8. Model Selection

The best-performing model was selected based on prediction accuracy and generalization performance.

### 9. Prediction

The final model was used to predict adview counts on unseen data.

---

## 📈 Key Learning Outcomes

During this internship project, I learned:

* Data preprocessing techniques
* Feature transformation and encoding
* Data visualization using Python
* Regression algorithms in machine learning
* Model comparison and evaluation
* Neural network implementation using Keras
* End-to-end machine learning workflow

---

## 📂 Repository Structure

```text
ML_project_on_YT_Adview_Prediction/
│
├── Dataset/
├── Notebook/
├── Model Files/
├── Results/
└── README.md
```

---

## 🎓 Internship Details

**Organization:** Internship Studio

**Internship Duration:** 27 April 2024 – 1 June 2024

**Domain:** Machine Learning

This project was completed as part of my Machine Learning internship program, where I applied machine learning concepts to solve a real-world regression problem.

---

## 👩‍💻 Author

**Usha Rani Villa**

* BCA Graduate
* MCA (Pursuing)
* Aspiring Data Analyst

GitHub: https://github.com/usharani003

---

⭐ If you found this project helpful, feel free to explore the repository.
