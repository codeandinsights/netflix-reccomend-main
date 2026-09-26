"""
Entry point for the CineIQ — Netflix Content Intelligence Flask application.
"""
import os
from app import create_app

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "false").lower() in ("true", "1")
    app.run(debug=debug, host="0.0.0.0", port=port)
