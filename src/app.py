from model.fashion import Model
from flask import Flask
from main import model

app = Flask(__name__)
app.register_blueprint(model.blue_model, url_prefix='/models')

if __name__=="__main__":
    app.run(host='0.0.0.0', port='5000', debug=True)