from datetime import datetime
from datetime import timedelta

from fastapi import APIRouter
from fastapi import Depends

from sqlalchemy.orm import Session

from jose import jwt

from passlib.context import CryptContext

from backend.app.core.database import get_db
from backend.app.models.user import User

router = APIRouter()

SECRET_KEY = "sonaris-secret-key"

ALGORITHM = "HS256"

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)


@router.post("/register")
def register(
    username: str,
    email: str,
    password: str,
    db: Session = Depends(get_db)
):
    existing = db.query(User).filter(
        User.username == username
    ).first()

    if existing:
        return {
            "error": "Username already exists"
        }

    user = User(
        username=username,
        email=email,
        password_hash=pwd_context.hash(password)
    )

    db.add(user)

    db.commit()

    db.refresh(user)

    return {
        "message": "User created",
        "user_id": user.id
    }


@router.post("/login")
def login(
    username: str,
    password: str,
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(
        User.username == username
    ).first()

    if not user:
        return {
            "error": "Invalid credentials"
        }

    if not pwd_context.verify(
        password,
        user.password_hash
    ):
        return {
            "error": "Invalid credentials"
        }

    token = jwt.encode(
        {
            "sub": user.username,
            "exp": datetime.utcnow() + timedelta(hours=12)
        },
        SECRET_KEY,
        algorithm=ALGORITHM
    )

    return {
        "access_token": token,
        "token_type": "bearer"
    }