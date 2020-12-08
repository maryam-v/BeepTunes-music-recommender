from flask import Flask
import os

def create_app(test_config=None):
    app = Flask(__name__,instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev'
    )
    if test_config is None:
        app.config.from_pyfile('config.py',silent=True)
    else:
        app.config.from_mapping(test_config)

    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    from . import db,recomm,entities
    db.init_app(app)
    app.register_blueprint(recomm.bp)
    app.register_blueprint(entities.bp)

    @app.route('/')
    def home():
        return 'beeptunes recommendation system.'
    return app
