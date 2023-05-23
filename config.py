import os
PROJECT_DIR= os.path.dirname(
    os.path.realpath( __file__)
)
TEMPLATE_PATH=os.path.join(PROJECT_DIR,"templates/image_template.html")
from jinja2 import Environment,FileSystemLoader, PackageLoader, select_autoescape
import re 
env = Environment(
    loader=FileSystemLoader(PROJECT_DIR),
    autoescape=select_autoescape()
)
env_template = env.get_template(TEMPLATE_PATH)
