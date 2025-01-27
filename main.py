from fastapi import FastAPI


from .services import notify
from .routes import task_routes, analysis_routes,auth_routes
from .db.database import create_table, SessionLocal
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler
import uvicorn


app = FastAPI()

# CORS setup
origins = ["http://localhost:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def start_up_event():
    create_table()
    
    #scheduler = BackgroundScheduler()
    #scheduler.add_job(notify.notify_due_tasks, args=[SessionLocal()], trigger="interval", seconds=10)
    #scheduler.start()



# Include routers
app.include_router(auth_routes.router, tags=["User Authentication"])
app.include_router(task_routes.router, prefix="/api/v1", tags=["Tasks CRUD"])
app.include_router(analysis_routes.analysis_routes, prefix="/api/v1", tags=["Analysis"])

# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)





















# from fastapi import FastAPI
# from . import controllers
# from .routes import task_routes,analysis_routes  # Assuming the import paths are correct
# from .database import create_table,SessionLocal
# from fastapi.middleware.cors import CORSMiddleware
# from apscheduler.schedulers.background import BackgroundScheduler
# from contextlib import asynccontextmanager


# db = SessionLocal()

# # Startup event to initialize the database table
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     create_table()



# scheduler = BackgroundScheduler()
# scheduler.add_job(controllers.notify_due_tasks,args=[db], trigger='interval',seconds=10)
# # scheduler.start()

# app = FastAPI(lifespan=lifespan)

# # CORS setup
# origins = ["http://localhost:8000"]
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )




# # Include routers
# app.include_router(task_routes.router, prefix="/api/v1",tags=["Tasks CRUD"])
# app.include_router(analysis_routes.analysis_routes, prefix="/api/v1",tags=["Analysis"])
