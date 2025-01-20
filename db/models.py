from sqlalchemy import Column, Integer, String, Text, DateTime 
from sqlalchemy.sql import func 
from .database import Base
import datetime

# status values -> ['pending','overdue','in-progress','completed']

class Task(Base):
    __tablename__ = "tasks"

    task_id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)  
    status = Column(String, default="pending",nullable=False) 
    due_date = Column(DateTime, nullable=False)
    completed_date = Column(DateTime, nullable=True)
    assigned_to = Column(String, nullable=False)
    priority = Column(String, nullable=False)   
    created_at = Column(DateTime, default=func.now(), nullable=False)


    def update_status(self, status: str):
        self.status = status
        if status == "completed":
            self.completed_date = func.now()  # datetime.now()

    def update_due_date(self, new_due_date: datetime.datetime):
        self.due_date = new_due_date


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

 

 




