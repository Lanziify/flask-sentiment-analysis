from flask import Flask
from .routes import index
from .config import cors


def create_app():
    app = Flask(__name__)
    cors.init_app(app)
    app.register_blueprint(index.index_blueprint)
    return app
