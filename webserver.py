from flask import Flask, render_template
from flask_restful import Api, Resource
 
app = Flask(__name__)
api = Api(app)

class HelloWorld(Resource):
    def get(self, X,Y):
        return {"X":X, "Y": Y}

    def post(self):
        return {"data": "Posted"}

api.add_resource(HelloWorld, "/<int:X>/<int:Y>")

if __name__ == "__main__":
    app.run(debug=True)