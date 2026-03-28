from dotenv import load_dotenv
from sqlalchemy import create_engine

# load the .env file variables
load_dotenv()


def db_connect():
    import os
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is not set. Add it to your .env file.")

    engine = create_engine(database_url)
    engine.connect()
    return engine
