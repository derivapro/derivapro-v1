import markdown
import os
from flask import Blueprint, render_template

index_bp = Blueprint('index', __name__)

@index_bp.route('/', methods=['GET'])
def index():
    
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'home.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    return render_template('index.html', md_content=md_content)

