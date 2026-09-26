"""
Flask application factory for CineIQ — Netflix Content Intelligence.
"""
from flask import Flask


def create_app():
    app = Flask(__name__)
    app.secret_key = "cineiq-dev-secret-key-2024"

    # Register blueprint
    from app.routes.main import main_bp
    app.register_blueprint(main_bp)

    return app


# Expose app instance for WSGI servers (e.g. gunicorn app:app)
app = create_app()
