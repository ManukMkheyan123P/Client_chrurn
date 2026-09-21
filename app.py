from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Загружаем модель и encoder
model = joblib.load('clientMLprediction.pkl')
ohe = joblib.load('churn_ohe.pkl')

cat_cols = ['contract_type', 'internet_type']


@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None
    probability = None

    if request.method == 'POST':

        # Получаем данные с сайта
        data = {
            'age': int(request.form['age']),
            'monthly_spend': float(request.form['monthly_spend']),
            'months_active': int(request.form['months_active']),
            'support_calls': int(request.form['support_calls']),
            'late_payments': int(request.form['late_payments']),
            'satisfaction': int(request.form['satisfaction']),
            'contract_type': request.form['contract_type'],
            'internet_type': request.form['internet_type'],
            'discount': float(request.form['discount'])
        }

        # Один клиент → DataFrame
        new_client = pd.DataFrame([data])

        # OneHotEncoding
        encoded = ohe.transform(new_client[cat_cols])

        # Удаляем исходные строковые столбцы
        new_client = new_client.drop(columns=cat_cols)

        # Добавляем закодированные столбцы
        new_client = new_client.join(encoded)

        # Prediction
        prediction = model.predict(new_client)[0]

        probability = model.predict_proba(new_client)[0][1]

        probability = round(probability * 100, 2)

    return render_template(
        'index.html',
        prediction=prediction,
        probability=probability
    )


if __name__ == '__main__':
    app.run(debug=True)