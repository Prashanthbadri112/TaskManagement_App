from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List

from ..services import notify

from ..db import models
from ..services import authentication
from .. import schemas
from ..db.database import get_db
import os
import ssl


# Initialize router
router = APIRouter()
ssl._create_default_https_context=ssl._create_unverified_context


# Get a task by ID
@router.get("/tasks/{task_id}", response_model=schemas.Task)
def get_task(task_id: int,current_user: models.User = Depends(authentication.get_current_user), db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found in the database")
    return db_task
    

# Get all tasks
@router.get("/all-tasks", response_model=List[schemas.Task])
def get_tasks(current_user: models.User = Depends(authentication.get_current_user),db: Session = Depends(get_db)):
    db_tasks = db.query(models.Task)
    if db_tasks is None:
        raise HTTPException(status_code=404, detail="Tasks not found")
    return db_tasks


# Create a new task
@router.post("/tasks/", response_model=schemas.Task)
def create_task(task: schemas.TaskCreate,current_user: models.User = Depends(authentication.get_current_user), db: Session = Depends(get_db)):
    try:
        if task.due_date <= datetime.now():
            raise HTTPException(status_code=400, detail="Due date must be greater than the current date")      
        db_task = models.Task(**task.dict())
        db.add(db_task)
        db.commit()
        db.refresh(db_task)
        return db_task
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        print(e)
        db.rollback()
        raise HTTPException(
            status_code=500,
            detail="An error occurred while creating the task."
        )



@router.put("/tasks/{task_id}", response_model=schemas.Task)
def update_task(task_id: int,task_update: schemas.TaskUpdate,current_user: models.User = Depends(authentication.get_current_user),  db: Session = Depends(get_db)):
    task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Update only provided fields
    for key, value in task_update.dict(exclude_unset=True).items():
        setattr(task, key, value)

        if key=='status':
            if value == 'completed':
                task.completed_date = datetime.now()
            else:
                task.completed_date = None
        if key=='due_date':
            if value.date() < task.created_at.date():
                task.status = 'overdue'
        # if key == 'due_date':
        #     print(f"Due date: {value}, Created at: {task.created_at}")  # Debugging log
        #     if value.date() < task.created_at.date():
        #         print("Setting status to overdue")  # Debugging log
        #         task.status = 'overdue'
    db.commit()
    db.refresh(task)
    return task


# Delete a task
@router.delete("/tasks/{task_id}", response_model=str)
def delete_task(task_id: int,current_user: models.User = Depends(authentication.get_current_user),  db: Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.task_id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    db.delete(db_task)
    db.commit()
    return "Deleted the task Successfully"

@router.get("/tasks/mail/{task_id}",response_model=str)
def mail(task_id:int,current_user: models.User = Depends(authentication.get_current_user), db:Session = Depends(get_db)):
    db_task = db.query(models.Task).filter(models.Task.task_id==task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    notify.email_notification(db_task)
    return "Successfully mailed the user"


# Create a new tasks bulky
# @router.post("/tasks/bulk", response_model=List[schemas.Task])
# def create_multiple_tasks(tasks: List[schemas.TaskCreate], db: Session = Depends(get_db)):
#     if not tasks:
#         raise HTTPException(status_code=400, detail="No tasks provided")
#     return task_controller.create_multiple_tasks(db=db, tasks=tasks)

# # Update task priority
# @router.put("/tasks/{task_id}/priority", response_model=schemas.Task)
# def update_task_priority(task_id: int, priority: schemas.TaskUpdatePriority, db: Session = Depends(get_db)):
#     db_task = controllers.update_task_priority(db=db, task_id=task_id, priority=priority)
#     if db_task is None:
#         raise HTTPException(status_code=404, detail="Task not found")
#     return db_task

# Update task status
# @router.put("/tasks/{task_id}/status", response_model=str)
# def update_task_status(task_id: int, status: schemas.TaskUpdateStatus, db: Session = Depends(get_db)):
#     db_task = task_controller.update_task_status(db=db, task_id=task_id, status=status)
#     if db_task is None:
#         raise HTTPException(status_code=404, detail="Task not found")
#     return db_task

# # Update task due date
# @router.put("/tasks/{task_id}/due-date", response_model=str)
# def update_task_due_date(task_id: int, due_date: schemas.TaskUpdateDueDate, db: Session = Depends(get_db)):
#     db_task = task_controller.update_task_due_date(db=db, task_id=task_id, due_date=due_date)
#     if db_task is None:
#         raise HTTPException(status_code=404, detail="Task not found")
#     return db_task