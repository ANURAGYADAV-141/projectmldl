# %%
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__)
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict():
    # receive the values send by user in three text boxes thru request object -> requesst.form.values()
    
    int_features = [x for x in request.form.values()]
    final_features = [np.array(int_features)]
    W=np.asarray(final_features).reshape(1,8)
    print(type(W))
    pred=model.predict(W)
    print(pred)
    if pred[0]==0:
        output="Not Diabetic"
        return render_template('index.html',p='Your diabetes result is :  {}'.format(output))
    else:
        output="Diabetic"
        return render_template('index.html',p='Your diabetes result is:  {}'.format(output))
    #if pred[0]==1:
     #   output="is"
     #   return render_template('index.html',p='Student passing probability is :  {}'.format(output))                           
 
    #prediction = model.predict(final_features)
    #output = round(prediction[0], 2)
    
    #prediction=model.predict_proba(final_features)
    #output='{0:.{1}f}'.format(prediction[0][1], 2)
   
    #print(output )

    #return render_template('index.html', pred='Student passing probability is :  {}'.format(output))

if __name__ == '__main__':
    app.run(debug=False)

# %%


# %%


# %%
