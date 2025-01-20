import pandas as pd
import io
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

def preprocessing(df):
    df['due_date'] = pd.to_datetime(df['due_date'],errors='coerce')
    df['completed_date'] = pd.to_datetime(df['completed_date'],errors='coerce')
    df['created_at'] = pd.to_datetime(df['created_at'],errors='coerce')


    #filling the null values with pending
    df['status'].fillna('pending')
    #df['completed_date'].fillna(pd.NaT)

    df_unique = df.drop_duplicates(subset=df.columns.difference(['task_id','created_at']))
    return df_unique

def task_completion_time(df):   
    df.dropna(subset=['completed_date'])
    df['completion_time'] = (df['completed_date'] - df['created_at']).dt.total_seconds() / 3600
    #df.loc[df['completion_time'].isna(), 'completion_time'] = 0
    return df

def identify_overdue_tasks(df):
    df['overdue'] = (df['completed_date'] > df['due_date'])
    return df

def priority_proportion(df):
    priority_dist = df['priority'].value_counts()
    return priority_dist

def analyze_task_completion_time(df):
    completed_tasks = df[df['status'] == 'completed']
    task_completion_time = completed_tasks['completion_time']
    return task_completion_time

def generate_csv_report(df):
    csv_data = df.to_csv(index=False)
    return csv_data

def plot_tasks_completed_per_day(df):
    df['completed_date'] = pd.to_datetime(df['completed_date'])
    df['completed_day'] = df['completed_date'].dt.date
    tasks_per_day = df.groupby('completed_day').size()

    fig = plt.figure(figsize=(10, 6))
    tasks_per_day.plot(kind='bar', color='skyblue')
    plt.title('Tasks Completed per Day', fontsize=16)
    plt.xlabel('Day', fontsize=12)
    plt.ylabel('Number of Tasks', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer,format='png')
    plt.close(fig)
    return img_buffer


def plot_task_distribution_by_priority(df):
    priority_counts = priority_proportion(df)

    fig = plt.figure(figsize=(8, 8))
    priority_counts.plot(kind='pie', autopct='%1.1f%%', startangle=140, colors=['gold', 'skyblue', 'lightcoral'])
    plt.title('Task Distribution by Priority', fontsize=16)
    plt.ylabel('')  # Remove default y-axis label
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer,format='png')
    plt.close(fig)
    return img_buffer
    

def plot_completion_trends(df):
    df['completed_date'] = pd.to_datetime(df['completed_date'])
    # completed_tasks = df.dropna(subset=['completed_date'])
    # completed_tasks['week'] = completed_tasks['completed_date'].dt.to_period('W').apply(str)
    # trends = completed_tasks.groupby('week').size()
    df['completed_day'] = df['completed_date'].dt.date
    trends = df.groupby('completed_day').size()
    fig = plt.figure(figsize=(10, 6))
    trends.plot(kind='line', marker='o', color='teal')
    plt.title('Task Completion Trends Over Time', fontsize=16)
    plt.xlabel('Week', fontsize=12)
    plt.ylabel('Number of Tasks Completed', fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer,format='png')
    plt.close(fig)
    return img_buffer

def plot_time_vs_priority(df):
    priority_order = {'low': 1, 'medium': 2, 'high': 3}
    df['priority_level'] = df['priority'].map(priority_order)

    fig = plt.figure(figsize=(8, 6))
    plt.scatter(df['priority_level'], df['completion_time'], alpha=0.7, c='orange')
    plt.title('Time to Complete Tasks vs. Task Priorities', fontsize=16)
    plt.xlabel('Priority Level (1=Low, 2=Medium, 3=High)', fontsize=12)
    plt.ylabel('Time to Complete Tasks (hours)', fontsize=12)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    plt.close(fig)
    return img_buf
