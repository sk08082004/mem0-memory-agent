## this give the project a central palce for configuration. 

import os 
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "Mem0 Long-Term Memory Agent"

USER_ID = os.getenv("USER_ID", "default_user")