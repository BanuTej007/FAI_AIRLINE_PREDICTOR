# FAI_AIRLINE_PREDICTOR

This project implements ML models such as Linear Regression, K Nearest Neighbors, Random Forest, and ANN to predict airline ticket prices based on various features like airline, source, destination, duration, and stops. The dataset used for this project is [https://www.kaggle.com/datasets/muhammadbinimran/flight-price-prediction](https://www.kaggle.com/datasets/muhammadbinimran/flight-price-prediction).

### Folder Structure
```markdown
├── __init__.py
├── preprocess.py
├── linear_reg.py
├── knn.py
├── random_forest.py
├── ann.py
├── Flight_prices_dataset.zip
├── requirements.txt
├── test.py
└── README.md
```

### Installation

1. Clone the repo in your local machine
```bash
git clone https://github.com/BanuTej007/FAI_AIRLINE_PREDICTOR
```

2. Install the required dependecies using the below commands
```bash
pip install -r requirements.txt
```

### Performance Results

| Model             | R²     | RMSE    | MAE     |
|-------------------|--------|---------|---------|
| Linear Regression | 0.4675 | 3356.20 | 2392.12 |
| KNN               | 0.7742 | 2167.03 | 1269.08 |
| __Random Forest__     | __0.8894__ | __1516.55__ | __750.15__  |
| ANN               | 0.8688 | 1651.46 | 963.56  |

### Author
Isha Pargaonkar

Banu Teja Maram

Abhishek Pramanik
