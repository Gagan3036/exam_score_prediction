from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        df = pd.read_csv(file)
        df = df.drop('Unnamed: 0', axis=1)
        df['TestPrep'] = df['TestPrep'].replace({'none':'0', 'completed':'1'}).astype(int)
        df['LunchType'] = df['LunchType'].replace({'standard':'0', 'free/reduced':'1'}).astype(int)
        df['ParentEduc'] = df['ParentEduc'].replace({
            "bachelor's degree":'1', 'some college':'2', "master's degree":'3',
            "associate's degree":'4', "high school":'5'}).astype(int)
        df['EthnicGroup'] = df['EthnicGroup'].replace({
            'group B':'0.25','group C':'0.50','group A':'0.75','group D':'1'}).astype(float)
        df['Gender'] = df['Gender'].replace({'female':'0','male':'1'}).astype(int)

        y = df['TestPrep']
        X = df[['Gender', 'MathScore', 'ReadingScore', 'WritingScore']]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=2524)

        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)

        return render_template('index.html', accuracy=accuracy, classification_report=class_report, confusion_matrix=conf_matrix)

if __name__ == '__main__':
    app.run(debug=True)
