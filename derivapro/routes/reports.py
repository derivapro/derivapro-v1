import markdown
import os
from flask import Blueprint, render_template

reports_generated_bp = Blueprint('reports_generated', __name__)

@reports_generated_bp.route('/reports-generated', methods=['GET'])
def reports_generated():
    
    readme_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports_generated.md')
    with open(readme_path, 'r') as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)
    
    return render_template('reports_generated.html', md_content=md_content)

