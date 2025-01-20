from pydantic import BaseModel
from datetime import datetime
from typing import Optional



# Schema for creating a task
class TaskCreate(BaseModel):
    name: str
    description: Optional[str] = None
    status: Optional[str] = "pending"
    due_date: datetime
    assigned_to: str
    priority:str
    created_at:Optional[datetime]=datetime.now()

    class Config:
        orm_mode = True  # Tells Pydantic to treat ORM models as dictionaries


# Schema for the task response (when fetching task data)
class Task(BaseModel):
    task_id: int
    name: str
    description: Optional[str] = None
    status: str
    due_date: datetime
    completed_date: Optional[datetime] = None
    assigned_to: str
    priority: str
    created_at: datetime

    class Config:
        orm_mode = True

# Schema for partial updates of task
class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    priority: Optional[str] = None

    class Config:
        orm_mode = True



class UserCreate(BaseModel):
    username: str
    password: str
 
    class Config:
        orm_mode = True




# # Schema for updating a task's status
# class TaskUpdateStatus(BaseModel):
#     #status: helper.Status
#     status: str

#     class Config:
#         orm_mode = True


# # Schema for updating a task's due date
# class TaskUpdateDueDate(BaseModel):
#     due_date: datetime

#     class Config:
#         orm_mode = True




