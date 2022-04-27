from fastapi import FastAPI
import uvicorn
import pickle
from uuid import uuid4

app = FastAPI(debug=True)

@app.get('/')
def index():
    return {'Status': 'All systems full power'}

@app.post('/predict')
def predict(input_text: list):
    """
    Función básica de predicción de twwets click-bait en inglés. Toma de
    entrada una lista de strings para generar un array con sus predicciones.

    :param input_text:
    :return:
    """

    # Cargar modelo para predicción
    model = pickle.load(open('../tfidf_lsvc.model', 'rb'))
    predictions = model.predict(input_text)

    # Si el resultado de la predicción no es una lista, lo convertimos
    if not isinstance(predictions, list):
        predictions = predictions.tolist()

    # Formateamos el resultado a un diccionario de diccionario
    data = format_response(predictions, input_text)
    return data

def format_response(predictions, input_text):
    # Prediction dict

    preds = {
        0: "Click Bait",
        1: "Real"
    }
    data = {}
    for p,t in zip(predictions, input_text):
        result = {
            "id": uuid4(),
            "input_text": t,
            "prediction": p,
            "predicted_class": preds[p]
        }
        data[len(data.values())] = result
    return data

if __name__  == '__main__':
    uvicorn.run(app)
