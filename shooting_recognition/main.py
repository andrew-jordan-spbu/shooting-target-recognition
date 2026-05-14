from dotenv import load_dotenv
load_dotenv()

from flask import Flask
from config.settings import DEBUG, PORT, SECRET_KEY
from database.db import init_db
from web.routes import bp


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static",
    )
    app.secret_key = SECRET_KEY
    app.register_blueprint(bp)
    return app


if __name__ == "__main__":
    init_db()
    app = create_app()
    print(f"Starting on http://localhost:{PORT}")
    app.run(debug=DEBUG, port=PORT, threaded=True)
