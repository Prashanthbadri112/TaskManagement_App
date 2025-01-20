import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail
from sqlalchemy.orm import Session
from ..db import models
from datetime import datetime, timedelta
from typing import List
from dotenv import load_dotenv

load_dotenv()
SENDGRID_API_KEY =os.getenv("SENDGRID_API_KEY","")
FROM_EMAIL = os.getenv("SENDER_EMAIL","")
SENDGRID_CLIENT = SendGridAPIClient(api_key=SENDGRID_API_KEY)


# Get all tasks
def get_tasks(db: Session):
    return db.query(models.Task)

def email_notification(task:models.Task):
    message = Mail(
        from_email= FROM_EMAIL,
        to_emails='prashanthbadri18@gmail.com',
        subject=f"Reminder: Task '{task.name}' is due soon!",
        html_content=f"""
        <html>
            <body>
                <h2>Hello {task.assigned_to},</h2>
                <p>This is a friendly reminder that your task:</p>
                <p><strong>{task.name}</strong></p>
                <p>is due on <strong>{task.due_date.strftime('%Y-%m-%d %H:%M')}</strong>.</p>
                <p>Details: {task.description}</p>
                <p><em>Please ensure the task is completed before the due date.</em></p>
                <p>Best Regards,<br>Task Management System</p>
            </body>
        </html>
        """
    )
    try:
        response = SENDGRID_CLIENT.send(message)
        print(f"Reminder sent for task: {task.name}")
    except Exception as e:
        print(f"Error sending email for task {task.name}: {e}")

def notify_due_tasks(db: Session):
    now = datetime.utcnow()
    one_hour_later = now+ timedelta(hours=24)

    tasks_due_soon = db.query(models.Task).filter(models.Task.due_date<=one_hour_later).all()

    for task in tasks_due_soon:
        email_notification(task)




# def create_multiple_tasks(db: Session, tasks: List[schemas.TaskCreate]):
#     created_tasks = [models.Task(**task.dict()) for task in tasks]
#     db.add_all(created_tasks)
#     db.commit()
#     return created_tasks