from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    print('test', flush=True)
    return "Hello world"


if __name__ == '__main__':
    app.run('localhost', port=6600)
