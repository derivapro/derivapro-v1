# derivapro/__init__.py
from app import create_app

def launch():
    app = create_app()
    app.run(debug=True)