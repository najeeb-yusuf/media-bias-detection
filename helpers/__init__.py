import sys
from config_handler import ConfigHandler
import os

try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yml')
    config = ConfigHandler.load_config(config_path)
except FileNotFoundError:
    print("'config.yml' not found")
    sys.exit(1)
except Exception as e:
    print(e)