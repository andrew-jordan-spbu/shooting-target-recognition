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
    import os
    init_db()
    app = create_app()
    # Bind to 0.0.0.0 so the port is reachable from outside the process (Docker).
    # Override via HOST env var if needed (e.g. HOST=127.0.0.1 for local-only).
    host = os.getenv("HOST", "0.0.0.0")
    print(f"Starting on http://{host}:{PORT}")
    app.run(host=host, debug=DEBUG, port=PORT, threaded=True)
