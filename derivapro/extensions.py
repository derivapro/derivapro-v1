from flask_bcrypt import Bcrypt
from flask_login import LoginManager
from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
migrate = Migrate()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
bcrypt = Bcrypt()


@login_manager.user_loader
def load_user(user_id):
    from .models.db_models import User

    return User.query.get(int(user_id))
