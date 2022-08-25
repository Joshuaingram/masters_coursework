"""
Assignment 3: Operationalizing Models
"""

"""

In this assignment we will be operationalizing the Decision Tree Classifier
by creating an API using fastapi and then containerizing it using Docker.

First we need to save the Decision Tree Classifier we created yesterday as
a pickle file. Go back to Assignment 2 and save your classifier after training it.

Example:

import pickle
from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()
dt.fit(train_X, train_y)
pickle.dump(dt, open('cereal_classifier_model', 'wb'))

Make sure that you make a copy of the saved model file in the same directory as
this Python file (app.py) so that we can easily load it.

"""

import pickle
import uvicorn
import pandas as pd
from fastapi import FastAPI, Form

app = FastAPI()

"""
If you named the model file something other than 'cereal_classifier_model',
you will need to update the line below accordingly.
"""
model = pickle.load(open('cereal_classifier_model', 'rb'))

"""
Warning: this function is going to look pretty ugly...
this isn't exactly how we would approach the problem in a real production
environment. However it is still a good learning opportunity.

Additionally, you may have to make some minor modifications based on
what features you used to train the Decision Tree Classifier and how you
encoded them.
"""
@app.post("/predict/cereal")
def predict_cereal(type: int = Form(),
            calories: int = Form(),
            protein: int = Form(),
            fat: int = Form(),
            sodium: int = Form(),
            fiber: int = Form(),
            carbo: int = Form(),
            sugars: int = Form(),
            potass: int = Form(),
            vitamins: int = Form(),
            shelf: int = Form(),
            weight: int = Form(),
            cups: int = Form(),
            A: int = Form(),
            G: int = Form(),
            K: int = Form(),
            N: int = Form(),
            P: int = Form(),
            Q: int = Form(),
            R: int = Form()
           ):
    columns = [
        'type',
        'calories',
        'protein',
        'fat',
        'sodium',
        'fiber',
        'carbo',
        'sugars',
        'potass',
        'vitamins',
        'shelf',
        'weight',
        'cups',
        'A',
        'G',
        'K',
        'N',
        'P',
        'Q',
        'R'
    ]
    df = pd.DataFrame([[
        type,
        calories,
        protein,
        fat,
        sodium,
        fiber,
        carbo,
        sugars,
        potass,
        vitamins,
        shelf,
        weight,
        cups,
        A,
        G,
        K,
        N,
        P,
        Q,
        R
    ]], columns=columns)
    
    pred = model.predict(df)
    return {
      "prediction": int(pred[0])
    }

"""
Now it's your turn...

Train a different classifier in a Jupyter notebook, save the model to a file,
and load it as we did above. Fill in an implementation for the function below
so that we can get predictions using the API.
"""
mymodel = pickle.load(open('loans_classifier_model', 'rb'))

@app.post("/predict/mymodel")
def predict_mymodel(
            last_fico_range_low: int = Form(),
            loan_amnt: int = Form()
           ):
    columns = [
        'last_fico_range_low',
        'loan_amnt'
    ]
    df = pd.DataFrame([[
        last_fico_range_low,
        loan_amnt
    ]], columns=columns)
    
    pred = mymodel.predict(df)
    return {
      "predicted interest rate": int(pred[0])
    }


"""
Finally, let's go ahead and containerize this API. A Dockerfile is already
provided for you. Feel free to review it and make sure you understand the steps.
When you are ready, run the following commands in your terminal (make sure your
terminal session is pointing towards the correct directory containing these files):

docker build -t model-api .

docker run -p 5000:5000 model-api
"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=4500)
