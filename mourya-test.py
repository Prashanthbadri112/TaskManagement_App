import time
import asyncio
import os
from apscheduler.schedulers.background import BackgroundScheduler
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Response
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
from pydantic import BaseModel
from datetime import datetime, timedelta
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from typing import List, Optional
import pandas as pd
import matplotlib.pyplot as plt
from fastapi.responses import FileResponse, StreamingResponse
import csv
import io
from dotenv import load_dotenv
import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
 
 
load_dotenv()
 
DATABASE_URL = "postgresql://postgres:root@localhost/dbtemp"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
 
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
SECRET_KEY = "helloMe110"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
 
class TaskModel(Base):
    __tablename__ = "tasks"
 
    task_id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    description = Column(Text, nullable=True)
    status = Column(String)
    due_date = Column(DateTime)
    assigned_to = Column(String)
    priority = Column(String)
    completed_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.now)
   
class Task(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "pending"
    due_date: datetime
    assigned_to: str
    priority: str
 
    class Config:
        orm_mode = True
 
class TaskUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    status: Optional[str] = None
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None
    priority: Optional[str] = None
 
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
 
class UserCreate(BaseModel):
    username: str
    password: str
 
    class Config:
        orm_mode = True
 
app = FastAPI()
 
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
 
 
 
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt
 
def verify_password(plain_password: str, hashed_password: str):
    return pwd_context.verify(plain_password, hashed_password)
 
def get_password_hash(password: str):
    return pwd_context.hash(password)
 
def decode_access_token(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.JWTError:
        return None
   
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    payload = decode_access_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
   
    db_user = db.query(User).filter(User.username == payload.get("sub")).first()
    if db_user is None:
        raise HTTPException(status_code=404, detail="User not found")
   
    return db_user
 
@app.post("/token/")
async def token(user: OAuth2PasswordRequestForm=Depends(), db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user is None or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
   
    access_token = create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}
 
@app.post("/register/")
async def create_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
   
    hashed_password = get_password_hash(user.password)
 
    new_user = User(username=user.username, hashed_password=hashed_password)
   
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
   
    return {"message": "User created successfully"}
 
@app.get("/")
@app.get("/tasks/")
async def get_all_tasks(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    tasks = db.query(TaskModel).all()
    return tasks
 
@app.get("/tasks/{task_id}")
async def get_task(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task
 
@app.post("/tasks/")
async def create_task(task: Task, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_task = TaskModel(**task.dict())
    db.add(db_task)
    db.commit()
    db.refresh(db_task)
    return db_task
 
@app.put("/tasks/{task_id}")
async def update_task(task_id: int, task: TaskUpdate, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
   
    update_data = task.dict(exclude_unset=True)
    if update_data.get('status') == 'completed':
        update_data['completed_date'] = datetime.now()
   
    for key, value in update_data.items():
        setattr(db_task, key, value)
   
    db.commit()
    db.refresh(db_task)
    return db_task
 
@app.delete("/tasks/{task_id}")
async def delete_task(task_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db_task = db.query(TaskModel).filter(TaskModel.task_id == task_id).first()
    if db_task is None:
        raise HTTPException(status_code=404, detail="Task not found")
   
    db.delete(db_task)
    db.commit()
    return {"message": "Task deleted successfully"}
 
@app.get("/download")
async def stream_csv(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    df = pd.read_sql(db.query(TaskModel).statement, db.bind)
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    buffer.seek(0)
   
    return StreamingResponse(
        buffer,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tasks.csv"}
    )
 
 
   
def cleaned_data(db: Session):
    df = pd.read_sql(db.query(TaskModel).statement, db.bind)
       
    df['due_date'] = pd.to_datetime(df['due_date'], errors='coerce')
    df['completed_date'] = pd.to_datetime(df['completed_date'], errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'])
 
    df = df.drop_duplicates(subset=['name', 'description', 'due_date', 'assigned_to', 'priority', 'status'])
 
    df['status'] = df['completed_date'].isna().replace({True: 'pending', False: 'completed'})
 
    return df.set_index('task_id').to_dict('list')
 
def format_duration(seconds):
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    parts = []
    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    return ", ".join(parts) if parts else "0 minutes"
 
@app.get("/stats")
async def process_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    df_dict = cleaned_data(db)
    df = pd.DataFrame(df_dict)
    completed_tasks = df[df['status'] == 'completed']
    completed_tasks['completion_time'] = (completed_tasks['completed_date'] - completed_tasks['created_at']).dt.total_seconds()
    avg_completion_time = format_duration(completed_tasks['completion_time'].mean())
    min_completion_time = format_duration(completed_tasks['completion_time'].min())
    max_completion_time = format_duration(completed_tasks['completion_time'].max())
    len_tasks = df.shape[0]
    len_pending_tasks = df[df['status'] == 'pending'].shape[0]
    len_inprogress_tasks = df[df['status'] == 'in-progress'].shape[0]
    len_completed_tasks = completed_tasks.shape[0]
    len_overdue_tasks = df[(df['due_date'] < pd.Timestamp.now()) & (df['status'] != 'completed')].shape[0]
   
    return {
        "total_tasks": len_tasks,
        "pending_tasks": len_pending_tasks,
        "inprogress_tasks": len_inprogress_tasks,
        "completed_tasks": len_completed_tasks,
        "overdue_tasks": len_overdue_tasks,
        "avg_completion_time": avg_completion_time,
        "min_completion_time": min_completion_time,
        "max_completion_time": max_completion_time
    }
 
 
@app.get("/graph/pie")
async def download_pie_chart(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    df = pd.read_sql(db.query(TaskModel.priority).statement, db.bind)
 
    priority_counts = df['priority'].value_counts()
    _, ax = plt.subplots()
    priority_counts.plot(kind='pie', autopct='%1.1f%%', ax = ax)
    ax.set_title('Task Distribution by Priority')
    ax.set_ylabel('')
   
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    return StreamingResponse(
        buffer,
        media_type='image/png',
        headers={"Content-Disposition": "attachment; filename=pie_chart.png"}
    )
 
@app.get("/graph/line")
async def download_line_chart(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        df_dict = cleaned_data(db)
        df = pd.DataFrame(df_dict)
        completion_trend = df[df['status'] == 'completed'].groupby(df['completed_date']).size()
        _, ax = plt.subplots(figsize=(12, 6))
        completion_trend.plot(kind='line', ax=ax)
        ax.set_title('Task Completion Trends Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Number of Tasks Completed')
       
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type='image/png',
            headers={"Content-Disposition": "attachment; filename=line_chart.png"}
        )
 
 
@app.get("/graph/bar")
async def download_bar_chart(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        df_dict = cleaned_data(db)
        df = pd.DataFrame(df_dict)
        _, ax = plt.subplots(figsize=(12, 6))
       
        df['completion_date'] = df['completed_date'].dt.date
        completed_per_day = df[df['status'] == 'completed'].groupby('completion_date').size()
        completed_per_day.plot(kind='bar', ax = ax)
        ax.set_title('Tasks Completed per Day')
        ax.set_xlabel('')
        ax.set_ylabel('Number of Tasks')
       
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type='image/png',
            headers={"Content-Disposition": "attachment; filename=bar_chart.png"}
        )
 
 
@app.get("/graph/scatter")
async def download_scatter_chart(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
        df_dict = cleaned_data(db)
        df = pd.DataFrame(df_dict)
        _, ax = plt.subplots(figsize=(12, 6))
 
        completed_tasks = df[df['status'] == 'completed']
        completed_tasks['completion_time'] = (completed_tasks['completed_date'] - completed_tasks['created_at']).dt.total_seconds() / 3600
        ax.scatter(completed_tasks['priority'], completed_tasks['completion_time'])
       
        ax.set_title('Completion Time vs Task Priority')
        ax.set_xlabel('Task Priority')
        ax.set_ylabel('Completion Time (hours)')
       
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        return StreamingResponse(
            buffer,
            media_type='image/png',
            headers={"Content-Disposition": "attachment; filename=scatter_chart.png"}
        )
 
 
def send_mail(subject, html_content):
    message = Mail(
        from_email='mouryapranay20@gmail.com',
        to_emails='mouryapranay20@gmail.com',
        subject=subject,
        html_content=html_content)
    try:
        sg = SendGridAPIClient(os.environ.get('SENDGRID_API_KEY'))
        response = sg.send(message)
    except Exception as e:
        print(e.message)
 
@app.on_event("startup")
def init_data(db: Session = Depends(get_db)):
    db = SessionLocal()
    try:
        df_dict = cleaned_data(db)
        df = pd.DataFrame(df_dict)
        len_pending_tasks = df[df['status'] == 'pending'].shape[0]
        len_inprogress_tasks = df[df['status'] == 'in-progress'].shape[0]
       
        if len_pending_tasks > 0 or len_inprogress_tasks > 0:
            subject = f"Task Status Update 🚀"
            html_content = f"""
            <html>
                <body>
                    <h1>Task Status Update</h1>
                    <p>Pending tasks: {len_pending_tasks} ⏳</p>
                    <p>In-progress tasks: {len_inprogress_tasks} 🔄</p>
                </body>
            </html>
            """
            scheduler = BackgroundScheduler()
            scheduler.add_job(send_mail, 'interval', days=1, args = [subject, html_content])
            scheduler.start()
       
    finally:
        db.close()
 
Base.metadata.create_all(bind=engine)
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
 
