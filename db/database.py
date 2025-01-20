from sqlalchemy import create_engine  # type: ignore
from sqlalchemy.ext.declarative import declarative_base  # type: ignore
from sqlalchemy.orm import sessionmaker, Session  # type: ignore

DATABASE_URL = "sqlite:///./task_management.db"

# Create engine and session maker
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Declare base for models
Base = declarative_base()

# Function to create all tables
def create_table():
    Base.metadata.create_all(bind=engine)

# New function to get DB session
def get_db() -> Session:
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()
