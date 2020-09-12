{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      " * Serving Flask app \"__main__\" (lazy loading)\n",
      " * Environment: production\n",
      "\u001b[31m   WARNING: This is a development server. Do not use it in a production deployment.\u001b[0m\n",
      "\u001b[2m   Use a production WSGI server instead.\u001b[0m\n",
      " * Debug mode: off\n"
     ]
    },
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      " * Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)\n",
      "127.0.0.1 - - [12/Sep/2020 21:22:39] \"\u001b[37mGET / HTTP/1.1\u001b[0m\" 200 -\n",
      "127.0.0.1 - - [12/Sep/2020 21:22:55] \"\u001b[37mPOST /predict HTTP/1.1\u001b[0m\" 200 -\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'numpy.ndarray'>\n",
      "[0]\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask,request, url_for, redirect, render_template\n",
    "import pickle\n",
    "import numpy as np\n",
    "\n",
    "app = Flask(__name__)\n",
    "model=pickle.load(open('model.pkl','rb'))\n",
    "\n",
    "@app.route('/')\n",
    "def home():\n",
    "    return render_template(\"index.html\")\n",
    "\n",
    "@app.route('/predict',methods=['POST','GET'])\n",
    "def predict():\n",
    "    # receive the values send by user in three text boxes thru request object -> requesst.form.values()\n",
    "    \n",
    "    int_features = [x for x in request.form.values()]\n",
    "    final_features = [np.array(int_features)]\n",
    "    W=np.asarray(final_features).reshape(1,8)\n",
    "    print(type(W))\n",
    "    pred=model.predict(W)\n",
    "    print(pred)\n",
    "    if pred[0]==0:\n",
    "        output=\"Not Diabetic\"\n",
    "        return render_template('index.html',p='Your diabetes result is :  {}'.format(output))\n",
    "    else:\n",
    "        output=\"Diabetic\"\n",
    "        return render_template('index.html',p='Your diabetes result is:  {}'.format(output))\n",
    "    #if pred[0]==1:\n",
    "     #   output=\"is\"\n",
    "     #   return render_template('index.html',p='Student passing probability is :  {}'.format(output))                           \n",
    " \n",
    "    #prediction = model.predict(final_features)\n",
    "    #output = round(prediction[0], 2)\n",
    "    \n",
    "    #prediction=model.predict_proba(final_features)\n",
    "    #output='{0:.{1}f}'.format(prediction[0][1], 2)\n",
    "   \n",
    "    #print(output )\n",
    "\n",
    "    #return render_template('index.html', pred='Student passing probability is :  {}'.format(output))\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    app.run(debug=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
