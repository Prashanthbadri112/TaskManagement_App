from fastapi import APIRouter,Depends,BackgroundTasks,Response
from fastapi.responses import StreamingResponse
from ..db import models,database
from ..services import analysis,authentication
from sqlalchemy.orm import Session
import pandas as pd
import io


analysis_routes = APIRouter()


@analysis_routes.get('/stats/tasks')
def get_stats(current_user: models.User = Depends(authentication.get_current_user),db: Session = Depends(database.get_db)):
    tasks = db.query(models.Task)
    df = pd.read_sql_query(tasks.statement,db.bind)
    df_process = analysis.preprocessing(df)
    df_complete = analysis.task_completion_time(df_process)
    desc_completion_time = df_complete['completion_time'].describe()
    overdue_tasks_count = len(df_complete[df_complete['completed_date']> df_complete['due_date']])
    pending_count = len(df_complete[df_complete['completed_date'].isna()])
    return {
        "Stats of Completion time":desc_completion_time,
        "NumberOfOverdueTasks": overdue_tasks_count,
        "PendingTasks": pending_count
    }
 
@analysis_routes.get('/download/report')
async def download_report(current_user: models.User = Depends(authentication.get_current_user), db: Session = Depends(database.get_db)):
    tasks = db.query(models.Task)
    df = pd.read_sql_query(tasks.statement,db.bind)
    df_process=analysis.preprocessing(df)
    df_complete = analysis.task_completion_time(df_process)
    print(df_complete)
    df_final = df_complete.iloc[:, 1:]
    df_csv = analysis.generate_csv_report(df_final)
    buffer = io.BytesIO()
    buffer.write(df_csv.encode())
    buffer.seek(0)
    
    headers = {
        'Content-Disposition': 'attachment; filename="report.csv"'
    }
    return StreamingResponse(buffer, media_type='text/csv', headers=headers)


@analysis_routes.get('/plot/{plot_type}')
def get_plots(plot_type:str,background_tasks:BackgroundTasks,current_user: models.User = Depends(authentication.get_current_user), db: Session = Depends(database.get_db)):
    tasks = db.query(models.Task)
    df = pd.read_sql_query(tasks.statement,db.bind)
    df_process=analysis.preprocessing(df)
    df_complete = analysis.task_completion_time(df_process)
    print(df_complete)
    if(plot_type=='line'):
        img_buffer = analysis.plot_completion_trends(df_complete)
    elif(plot_type=='pie'):
        img_buffer = analysis.plot_task_distribution_by_priority(df_complete)
    elif(plot_type=='bar'):
        img_buffer = analysis.plot_tasks_completed_per_day(df_complete)
    elif(plot_type=="scatter"):
        img_buffer=analysis.plot_time_vs_priority(df_complete)

    
    buffer_info : bytes = img_buffer.getvalue()
    background_tasks.add_task(img_buffer.close)
    headers = {'Content-Disposition': 'inline; filename="out.png"'}
    return Response(buffer_info,headers=headers,media_type='image/png')

