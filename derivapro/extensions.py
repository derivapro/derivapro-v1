import os

from flask_bcrypt import Bcrypt
from flask_caching import Cache
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import CSRFProtect


class LazyMigrate:
    """Load Flask-Migrate only when migration commands are explicitly needed."""

    def __init__(self):
        self._migrate = None

    def init_app(self, app, db):
        enabled = os.getenv("ENABLE_FLASK_MIGRATE", "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            app.logger.debug(
                "Flask-Migrate is disabled for normal app startup. "
                "Set ENABLE_FLASK_MIGRATE=1 to enable migration commands."
            )
            return None

        from flask_migrate import Migrate

        self._migrate = Migrate()
        return self._migrate.init_app(app, db)

# Initialize Flask extensions
db = SQLAlchemy()
migrate = LazyMigrate()
login_manager = LoginManager()
login_manager.login_view = "auth.login"
bcrypt = Bcrypt()
csrf = CSRFProtect()
cache = Cache()


@login_manager.user_loader
def load_user(user_id):
    from .models.db_models import User

    return User.query.get(int(user_id))
