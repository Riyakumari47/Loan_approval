from django.shortcuts import render, redirect
from django.conf import settings
from pathlib import Path
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from django.template.defaulttags import register

_trained_model = None
_trained_scaler = None
_label_encoders = {}

@register.filter
def keyvalue(dict_data, key):
    return dict_data.get(key)

def home(request):
    return render(request, 'pipeline/home.html')

def upload_dataset(request):
    message = ''
    if request.method == 'POST' and request.FILES.get('dataset'):
        uploaded_file = request.FILES['dataset']

        if not uploaded_file.name.lower().endswith(('.csv', '.xlsx')):
            message = 'Please upload a CSV or Excel file.'
        else:
            dataset_dir = Path(settings.MEDIA_ROOT) / 'datasets'
            dataset_dir.mkdir(parents=True, exist_ok=True)
            file_path = dataset_dir / Path(uploaded_file.name).name

            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)

            try:
                df = pd.read_csv(file_path) if file_path.suffix.lower() == '.csv' else pd.read_excel(file_path)
                request.session['dataset_path'] = str(file_path)
                return redirect('analysis')
            except Exception as error:
                message = f'Could not read the dataset: {error}'

    return render(request, 'pipeline/upload.html', {'message': message})

def analysis(request):
    global _trained_model, _trained_scaler, _label_encoders
    file_path = request.session.get('dataset_path')
    if not file_path or not Path(file_path).exists():
        return render(request, 'pipeline/upload.html', {'message': 'Please upload a dataset first.'})

    try:
        path = Path(file_path)
        df = pd.read_csv(path) if path.suffix.lower() == '.csv' else pd.read_excel(path)
        df = df.drop_duplicates()

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            df[col] = np.clip(df[col], q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')

        plt.figure(figsize=(8, 5))
        if len(num_cols) > 1:
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        else:
            plt.text(0.5, 0.5, "Not enough numeric features for heatmap", ha='center', va='center')
        
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        heatmap_png = base64.b64encode(buffer.getvalue()).decode('utf-8')
        buffer.close()
        plt.close()

        target_col = df.columns[-1]
        X = df.drop(columns=[target_col])
        y = df[target_col]

        _label_encoders = {}
        for col in X.select_dtypes(exclude=[np.number]).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            _label_encoders[col] = le

        target_le = LabelEncoder()
        if y.dtype == 'object':
            y = target_le.fit_transform(y.astype(str))
            _label_encoders['__target__'] = target_le

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        _trained_scaler = StandardScaler()
        X_train_scaled = _trained_scaler.fit_transform(X_train)
        X_test_scaled = _trained_scaler.transform(X_test)

        _trained_model = RandomForestClassifier(n_estimators=100, random_state=42)
        _trained_model.fit(X_train_scaled, y_train)
        predictions = _trained_model.predict(X_test_scaled)

        metrics = {
            'accuracy': round(accuracy_score(y_test, predictions) * 100, 2),
            'precision': round(precision_score(y_test, predictions, average='macro', zero_division=0) * 100, 2),
            'recall': round(recall_score(y_test, predictions, average='macro', zero_division=0) * 100, 2),
            'f1': round(f1_score(y_test, predictions, average='macro', zero_division=0) * 100, 2),
        }

        request.session['features'] = X.columns.tolist()

        return render(request, 'pipeline/analysis.html', {
            'columns': df.columns.tolist(),
            'rows': df.head(10).to_dict('records'),
            'row_count': len(df),
            'column_count': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'chart_heatmap': heatmap_png,
            'metrics': metrics,
            'features': X.columns.tolist()
        })
    except Exception as error:
        return render(request, 'pipeline/upload.html', {'message': f'Pipeline Processing Error: {error}'})

def predict(request):
    global _trained_model, _trained_scaler, _label_encoders
    features = request.session.get('features', [])
    prediction_result = None

    if request.method == 'POST':
        inputs = []
        for feat in features:
            val = request.POST.get(feat, 0)
            try:
                inputs.append(float(val))
            except ValueError:
                inputs.append(0.0)
        
        if _trained_model and _trained_scaler and len(inputs) == len(features):
            try:
                input_array = np.array([inputs])
                input_scaled = _trained_scaler.transform(input_array)
                pred_class = _trained_model.predict(input_scaled)[0]
                
                if '__target__' in _label_encoders:
                    prediction_result = _label_encoders['__target__'].inverse_transform([pred_class])[0]
                else:
                    prediction_result = "Approved / Positive" if pred_class == 1 else "Rejected / Negative"
            except Exception as e:
                prediction_result = f"Error during prediction: {e}"
        else:
            prediction_result = "Model is not trained yet or input mismatch. Please re-upload dataset."

    return render(request, 'pipeline/prediction.html', {
        'features': features,
        'prediction_result': prediction_result
    })