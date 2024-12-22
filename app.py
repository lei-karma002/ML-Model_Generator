from flask import Flask, render_template, request, jsonify, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression, make_blobs
import joblib
import io
import zipfile
import matplotlib
matplotlib.use('Agg')  # Set the backend to Agg
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import json
import os
import math
from scipy import stats
import base64
import io
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, request, jsonify
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.cluster import KMeans
from io import BytesIO
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, r2_score, mean_squared_error, 
    mean_absolute_error
)
from scipy import stats


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate_data', methods=['POST'])
def generate_synthetic_data():
    try:
        data_type = request.form['dataType']
        n_features = int(request.form['nFeatures'])

        if data_type == 'classification':
            # Get class names and their settings
            class_names = request.form.getlist('className[]')
            class_samples = [int(x) for x in request.form.getlist('classSamples[]')]
            
            # Total number of samples
            n_samples = sum(class_samples)
            n_classes = len(class_names)
            
            # Calculate minimum number of informative features needed
            min_informative = math.ceil(math.log2(n_classes * 2))
            n_informative = max(min_informative, n_features - 2)
            
            # Generate data with numeric labels first
            X, y_numeric = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_classes=n_classes,
                n_informative=n_informative,
                n_redundant=max(0, n_features - n_informative),
                weights=[s/n_samples for s in class_samples],
                random_state=42
            )
            
            # Create class mapping and convert numeric to string labels
            class_mapping = {i: name for i, name in enumerate(class_names)}
            y = [class_mapping[int(label)] for label in y_numeric]
            
            # Save both numeric mapping and string mapping
            numeric_mapping = {name: i for i, name in class_mapping.items()}
            
            # Convert to DataFrame
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            
            # Save numeric labels for model training
            df['target_numeric'] = y_numeric
            # Save string labels for display
            df['target'] = y

            # Save data and mappings
            df.to_csv('synthetic_data.csv', index=False)
            mappings = {
                'class_mapping': class_mapping,
                'numeric_mapping': numeric_mapping
            }
            with open('class_mapping.json', 'w') as f:
                json.dump(mappings, f)

        else:  # regression
            n_samples = int(request.form['nSamples'])
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                noise=0.1,
                random_state=42
            )
            
            # Convert to DataFrame
            feature_names = [f'feature_{i+1}' for i in range(n_features)]
            df = pd.DataFrame(X, columns=feature_names)
            df['target'] = y
            
            # Save data
            df.to_csv('synthetic_data.csv', index=False)

        return jsonify({
            'success': True,
            'message': f'Synthetic {data_type} data generated successfully',
            'features': feature_names,
            'samples': n_samples if data_type == 'regression' else None,
            'classNames': class_names if data_type == 'classification' else None
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/train', methods=['POST'])
def train_model():
    try:
        algorithm = request.form['algorithm']
        train_size = float(request.form['trainSize']) / 100
        
        # Load the synthetic data
        df = pd.read_csv('synthetic_data.csv')
        
        # Check if it's classification or regression
        is_classification = 'target_numeric' in df.columns
        is_regression = algorithm == 'linear_regression'  
        
        if is_classification:
            X = df.drop(['target', 'target_numeric'], axis=1)
            y = df['target_numeric']
        else:  # regression
            X = df.drop(['target'], axis=1)
            y = df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_size,
            random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model based on selected algorithm and type
        if is_classification:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            joblib.dump(model, 'model.joblib')
            joblib.dump(scaler, 'scaler.joblib')
            
            return jsonify({
                'success': True,
                'train_accuracy': model.score(X_train_scaled, y_train),
                'test_accuracy': model.score(X_test_scaled, y_test),
                'metric_type': 'accuracy'
            })
        else:  # regression
            model = LinearRegression()
            model.fit(X_train_scaled, y_train)
            
            # Save model and scaler
            joblib.dump(model, 'model.joblib')
            joblib.dump(scaler, 'scaler.joblib')
            
            return jsonify({
                'success': True,
                'train_r2': model.score(X_train_scaled, y_train),
                'test_r2': model.score(X_test_scaled, y_test),
                'metric_type': 'r2'
            })
        
    except Exception as e:
        print("Training error:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/simulate', methods=['POST'])
def simulate():
    try:
        # Load the trained model and scaler
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Get simulation parameters
        n_simulations = int(request.form['n_simulations'])
        
        # Fix the feature values handling
        feature_values = []
        for i in range(1, len(request.form)):  # Start from 1 since we have n_simulations
            key = f'feature_{i}'
            if key in request.form:
                feature_values.append(float(request.form[key]))
        
        # Check if it's classification or regression
        is_classification = isinstance(model, RandomForestClassifier)
        
        # Create base input array
        X = np.array([feature_values])
        X_scaled = scaler.transform(X)
        
        # Run simulations
        simulation_results = []
        
        for _ in range(n_simulations):
            if is_classification:
                # For classification, use predict_proba to get class probabilities
                probs = model.predict_proba(X_scaled)[0]
                # Simulate outcome based on probabilities
                prediction = np.random.choice(model.classes_, p=probs)
                simulation_results.append(prediction)
            else:
                # For regression, add noise to the prediction
                base_prediction = model.predict(X_scaled)[0]
                # Add random noise based on model's residual standard error
                noise = np.random.normal(0, 0.1 * np.abs(base_prediction))  # 10% noise
                simulation_results.append(base_prediction + noise)
        
        # Calculate simulation statistics
        if is_classification:
            # Load class mapping
            with open('class_mapping.json', 'r') as f:
                mappings = json.load(f)
            class_mapping = mappings['class_mapping']
            
            # Convert numeric predictions to class names
            simulation_results = [class_mapping[str(int(x))] for x in simulation_results]
            
            # Calculate class probabilities
            unique_classes = np.unique(simulation_results)
            class_counts = {cls: simulation_results.count(cls) for cls in unique_classes}
            total = len(simulation_results)
            probabilities = {cls: count/total for cls, count in class_counts.items()}
            
            stats_result = {
                'class_probabilities': probabilities,
                'most_likely_class': max(probabilities.items(), key=lambda x: x[1])[0]
            }
        else:
            # For regression, calculate statistical measures
            stats_result = {
                'mean': float(np.mean(simulation_results)),
                'std': float(np.std(simulation_results)),
                'ci_lower': float(np.percentile(simulation_results, 2.5)),
                'ci_upper': float(np.percentile(simulation_results, 97.5)),
                'min': float(np.min(simulation_results)),
                'max': float(np.max(simulation_results))
            }
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        if is_classification:
            # Bar plot for classification
            classes = list(probabilities.keys())
            probs = list(probabilities.values())
            plt.bar(classes, probs)
            plt.title('Simulation Results: Class Probabilities')
            plt.ylabel('Probability')
            plt.xticks(rotation=45)
        else:
            # Histogram for regression
            plt.hist(simulation_results, bins=30, density=True)
            plt.title('Simulation Results Distribution')
            plt.xlabel('Predicted Value')
            plt.ylabel('Density')
            
            # Add normal distribution curve
            mu = np.mean(simulation_results)
            sigma = np.std(simulation_results)
            x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
            plt.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', lw=2)
            
        plt.tight_layout()
        simulation_plot = get_plot_as_base64()
        plt.close()
        
        return jsonify({
            'success': True,
            'statistics': stats_result,
            'plot': simulation_plot
        })
        
    except Exception as e:
        print("Simulation error:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/download_model')
def download_model():
    try:
        memory_file = BytesIO()
        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            if os.path.exists('model.joblib'):
                zf.write('model.joblib')
            else:
                raise FileNotFoundError("Model file not found. Please train the model first.")
            
            if os.path.exists('scaler.joblib'):
                zf.write('scaler.joblib')
            
            if os.path.exists('class_mapping.json'):
                zf.write('class_mapping.json')
            
            readme_content = """Model Package Contents:
1. model.joblib - The trained model
2. scaler.joblib - The fitted StandardScaler
3. class_mapping.json - (For classification only) Class label mappings

Usage:
import joblib
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# For making predictions:
X_scaled = scaler.transform(X)
predictions = model.predict(X_scaled)
"""
            zf.writestr('README.txt', readme_content)
        
        memory_file.seek(0)
        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='model_package.zip'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/get_eda', methods=['GET'])
def get_eda():
    try:
        if not os.path.exists('synthetic_data.csv'):
            raise FileNotFoundError("Synthetic data file not found")
            
        df = pd.read_csv('synthetic_data.csv')
        is_classification = 'target_numeric' in df.columns
        eda_results = {}
        
        if is_classification:
            df_numeric = df.drop(['target', 'target_numeric'], axis=1)
            eda_results['basic_stats'] = df_numeric.describe().to_html()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            eda_results['correlation_plot'] = get_plot_as_base64()
            plt.close()
            
            n_features = len(df_numeric.columns)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            plt.figure(figsize=(5 * n_cols, 4 * n_rows))
            for i, column in enumerate(df_numeric.columns, 1):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(df_numeric[column], kde=True)
                plt.title(f'{column} Distribution')
            plt.tight_layout(pad=3.0)
            eda_results['distribution_plot'] = get_plot_as_base64()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            target_counts = df['target'].value_counts()
            sns.barplot(x=target_counts.index, y=target_counts.values)
            plt.title('Target Distribution')
            plt.xlabel('Class')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.tight_layout()
            eda_results['target_plot'] = get_plot_as_base64()
            plt.close()
            
            n_features_to_plot = min(3, n_features)
            plt.figure(figsize=(5 * n_features_to_plot, 5))
            for i, column in enumerate(df_numeric.columns[:n_features_to_plot], 1):
                plt.subplot(1, n_features_to_plot, i)
                sns.boxplot(data=df, x='target', y=column)
                plt.title(f'{column} by Class')
                plt.xticks(rotation=45)
            plt.tight_layout(pad=2.0)
            eda_results['scatter_plot'] = get_plot_as_base64()
            plt.close()
            
        else:  # Regression
            df_features = df.drop('target', axis=1)
            n_features = len(df_features.columns)
            n_cols = min(3, n_features)
            n_rows = (n_features + n_cols - 1) // n_cols
            
            eda_results['basic_stats'] = df.describe().to_html()
            
            plt.figure(figsize=(max(10, n_features + 1), max(8, n_features + 1)))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix (including Target)')
            plt.tight_layout()
            eda_results['correlation_plot'] = get_plot_as_base64()
            plt.close()
            
            plt.figure(figsize=(5 * n_cols, 4 * n_rows))
            for i, column in enumerate(df_features.columns, 1):
                plt.subplot(n_rows, n_cols, i)
                sns.histplot(df[column], kde=True)
                plt.title(f'{column} Distribution')
            plt.tight_layout(pad=3.0)
            eda_results['distribution_plot'] = get_plot_as_base64()
            plt.close()
            
            plt.figure(figsize=(10, 6))
            sns.histplot(df['target'], kde=True)
            plt.title('Target Distribution')
            plt.xlabel('Target Value')
            plt.ylabel('Count')
            plt.tight_layout()
            eda_results['target_plot'] = get_plot_as_base64()
            plt.close()
            
            n_features_to_plot = min(3, n_features)
            plt.figure(figsize=(5 * n_features_to_plot, 5))
            for i, column in enumerate(df_features.columns[:n_features_to_plot], 1):
                plt.subplot(1, n_features_to_plot, i)
                sns.scatterplot(data=df, x=column, y='target')
                plt.title(f'{column} vs Target')
            plt.tight_layout(pad=2.0)
            eda_results['scatter_plot'] = get_plot_as_base64()
            plt.close()
            
            eda_results['skewness'] = df.skew().to_dict()
            eda_results['kurtosis'] = df.kurtosis().to_dict()
        
        return jsonify(eda_results)
        
    except Exception as e:
        print("Error in get_eda:", str(e))
        return jsonify({'error': str(e)})

def get_plot_as_base64():
    buffer = BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    return base64.b64encode(image_png).decode('utf-8')

# Dummy true and predicted values (replace with your actual model predictions)
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]

# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

# Function to generate the confusion matrix plot
def generate_confusion_matrix_plot(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64

# Function to generate the residual plot (example)
def generate_residual_plot(y_true, y_pred):
    residuals = np.array(y_true) - np.array(y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_pred, residuals)
    ax.axhline(0, color='red', linestyle='--')
    ax.set_xlabel('Predicted values')
    ax.set_ylabel('Residuals')
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64

# Route to evaluate the model
@app.route('/evaluate_model', methods=['GET'])
def evaluate_model():
    try:
        # Load original and simulated data
        df = pd.read_csv('synthetic_data.csv')
        model = joblib.load('model.joblib')
        scaler = joblib.load('scaler.joblib')
        
        # Determine if classification or regression
        is_classification = 'target_numeric' in df.columns
        
        if is_classification:
            X = df.drop(['target', 'target_numeric'], axis=1)
            y_true = df['target_numeric']
        else:
            X = df.drop(['target'], axis=1)
            y_true = df['target']
            
        # Scale features
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        
        # Generate evaluation results
        results = {}
        
        if is_classification:
            # Classification metrics
            results['metrics'] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted'),
                'recall': recall_score(y_true, y_pred, average='weighted'),
                'f1_score': f1_score(y_true, y_pred, average='weighted')
            }
            
            # Confusion Matrix
            plt.figure(figsize=(8, 6))
            cm = confusion_matrix(y_true, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            results['confusion_matrix'] = get_plot_as_base64()
            plt.close()
            
            # ROC Curve
            plt.figure(figsize=(8, 6))
            y_prob = model.predict_proba(X_scaled)
            for i, class_name in enumerate(model.classes_):
                fpr, tpr, _ = roc_curve(y_true == i, y_prob[:, i])
                plt.plot(fpr, tpr, label=f'Class {class_name}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            results['roc_curve'] = get_plot_as_base64()
            plt.close()
            
        else:
            # Regression metrics
            results['metrics'] = {
                'r2_score': r2_score(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mae': mean_absolute_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred))
            }
            
            # Residual Plot
            plt.figure(figsize=(8, 6))
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residual Plot')
            results['residual_plot'] = get_plot_as_base64()
            plt.close()
            
            # Q-Q Plot
            plt.figure(figsize=(8, 6))
            stats.probplot(residuals, dist="norm", plot=plt)
            plt.title('Q-Q Plot')
            results['qq_plot'] = get_plot_as_base64()
            plt.close()
            
            # Actual vs Predicted
            plt.figure(figsize=(8, 6))
            plt.scatter(y_true, y_pred, alpha=0.5)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            results['actual_vs_predicted'] = get_plot_as_base64()
            plt.close()
        
        return jsonify({
            'success': True,
            'is_classification': is_classification,
            **results
        })
    
    except Exception as e:
        print("Evaluation error:", str(e))
        return jsonify({
            'success': False,
            'error': str(e)
        })
if __name__ == '__main__':
    app.run(debug=True) 