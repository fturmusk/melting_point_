📘 Predicting Road Accident Risk
Kaggle Playground Series – Season 5 Episode 10


🧩 Overview
This project is part of the Kaggle Playground Series S5E10, where the goal is to predict the likelihood of accidents occurring on various types of roads. The competition provides a structured dataset simulating real‑world road and traffic conditions, and the objective is to build a regression model that minimizes the Root Mean Squared Error (RMSE) between predicted and observed accident risk values.

🔗 Competition link:
https://www.kaggle.com/competitions/playground-series-s5e10


🎯 Objective
The task is to:
 -Analyze the dataset (EDA)
 -Preprocess the features (encoding, scaling, cleaning)
 -Train machine learning models capable of risk prediction
 -Evaluate predictions using RMSE, the official metric
 -Generate a valid Kaggle submission file


📂 Project Structure

```
road_accident_risk/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
│
├── src/
│   ├── preprocess.py        # Data loading, cleaning, encoding, scaling
│   ├── model.py             # Model training, evaluation, tuning
│
├── main.py                  # Main execution pipeline
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

▶️ Running the Project
```
pip install -r requirements.txt
python main.py
```



