from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ..db.database import get_db
from ..db import models
from ..services import authentication
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from .. import schemas

 
router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@router.post("/token/")
async def token(user: OAuth2PasswordRequestForm=Depends(), db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user is None or not authentication.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid username or password")
   
    access_token = authentication.create_access_token(data={"sub": db_user.username})
    return {"access_token": access_token, "token_type": "bearer"}



@router.post("/register/")
async def create_user(user: schemas.UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Username already exists")
   
    hashed_password = authentication.get_password_hash(user.password)
 
    new_user = models.User(username=user.username, hashed_password=hashed_password)
   
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
   
    return {"message": "User created successfully"}