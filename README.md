# Laptop Price Predictor

A Flask-based machine learning web application that predicts laptop prices based on technical specifications such as brand, processor, RAM, storage, screen details, GPU, and operating system.

## Live Application

Run the project locally at:

```text
http://127.0.0.1:5000
```

## Features

- Predicts laptop prices from selected specifications
- Uses a trained Random Forest machine learning model
- Provides a simple, user-friendly web interface
- Includes laptop datasets for model training and analysis
- Includes comparison, prediction, and information pages

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- HTML
- CSS

## Project Structure

```text
laptop-price-predictor/
│
├── app.py                    # Main Flask application
├── train_model.py            # Script for training the ML model
├── view_columns.py           # Displays dataset/model columns
├── rf_model.pkl              # Trained Random Forest model
├── columns.pkl               # Saved input feature columns
├── dataset1.csv              # Laptop dataset
├── comparison_dataset.csv    # Dataset used for comparison
├── requirements.txt          # Required Python libraries
├── Procfile                  # Deployment configuration
│
├── static/                   # CSS, images, and static assets
└── templates/                # HTML templates
    ├── index.html
    ├── prediction.html
    ├── comparison.html
    └── news.html
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/udayk2003-dot/laptop-price-predictor.git
```

### 2. Move into the project folder

```bash
cd laptop-price-predictor
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Install the required packages

Windows:

```bash
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 5. Run the application

```bash
python app.py
```

### 6. Open the application

Open the following address in your browser:

```text
http://127.0.0.1:5000
```

## How It Works

1. The user enters laptop specifications on the prediction page.
2. The application processes the selected specifications.
3. The trained Random Forest model predicts the estimated laptop price.
4. The predicted result is displayed in the web application.

## Author

**Uday K**  
GitHub: [udayk2003-dot](https://github.com/udayk2003-dot)

## Repository

[View the source code](https://github.com/udayk2003-dot/laptop-price-predictor)
