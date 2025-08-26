from flask import Flask, request, render_template
import numpy as np
import pickle

# importing model and scaler
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))  # only minmax scaler is used now

# creating flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # put features in list
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    # scale features only with minmax
    final_features = ms.transform(single_pred)

    # predict probabilities
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(final_features)[0]
        top_indices = np.argsort(probabilities)[-3:][::-1]  # top 3
    else:
        # fallback if model doesn't support predict_proba
        top_indices = [model.predict(final_features)[0]]

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
        7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
        12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
        21: "Chickpea", 22: "Coffee"
    }

    # build recommendation list
    recommendations = []
    for idx in top_indices:
        if idx in crop_dict:
            recommendations.append(crop_dict[idx])

    if recommendations:
        result = "Top recommended crops: " + ", ".join(recommendations)
    else:
        result = "Sorry, we could not determine the best crops to be cultivated with the provided data."

    return render_template('index.html', result=result)


# python main
if __name__ == "__main__":
    app.run(debug=True)
