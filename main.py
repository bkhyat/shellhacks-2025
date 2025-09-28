# main.py

import os
import json
import re
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware  # 1. IMPORT THE MIDDLEWARE
from pydantic import BaseModel, EmailStr
from typing import Optional, AsyncGenerator, List, Dict, Any
from contextlib import asynccontextmanager
from datetime import datetime, timedelta

# Environment & AI
from dotenv import load_dotenv
import google.generativeai as genai
# from google import genai

# Security & Auth
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext
from jose import JWTError, jwt

# Database
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, selectinload
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload
from models import Base, User, Topic, SubTopic, LearningResource, ProficiencyLevel, Quiz, Question, Option

# --- Initial Configuration ---
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

DATABASE_URL = "sqlite+aiosqlite:///./test.db"
engine = create_async_engine(DATABASE_URL)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
MAX_BCRYPT_LEN = 72

# --- Application Lifespan ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(lifespan=lifespan, title="Learning Plan API")

# 2. ADD THE CORS MIDDLEWARE

origins = [
    # "http://localhost.tiangolo.com",
    # "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- END OF CORS CONFIGURATION ---


# --- Security & Auth Configuration ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = os.getenv("SECRET_KEY", "a_super_secret_dev_key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")


# --- Pydantic Schemas ---
# Schemas for API input and DB-based output remain for other endpoints
class UserSignUp(
    BaseModel): name: str; email: EmailStr; password: str; confirm_password: str; birth_year: int; school_year: str; major: str


class UserLogin(BaseModel): email: EmailStr; password: str


class Token(BaseModel): access_token: str; token_type: str


class TopicCreate(BaseModel):
    topic_name: str
    age: Optional[int] = None
    education: Optional[str] = None
    major: Optional[str] = None


class LearningResourceDisplay(BaseModel):
    id: int;
    type: str;
    title: str;
    url: str

    class Config: from_attributes = True


# class SubTopicDisplay(BaseModel):
#     id: int;
#     sub_topic_name: str;
#     duration: str;
#     proficiency: ProficiencyLevel
#     order: int;
#     is_complete: bool;
#     resources: List[LearningResourceDisplay] = []
#
#     class Config: from_attributes = True


class OptionDisplay(BaseModel):
    id: int;
    option_text: str

    class Config: from_attributes = True


class QuestionDisplay(BaseModel):
    id: int;
    question_text: str;
    correct_option_index: int
    options: List[OptionDisplay]

    class Config: from_attributes = True


class QuizDisplay(BaseModel):
    id: int
    questions: List[QuestionDisplay]

    class Config: from_attributes = True


# Add the new QuizDisplay model to the SubTopicDisplay
class SubTopicDisplay(BaseModel):
    id: int;
    sub_topic_name: str;
    duration: str;
    proficiency: ProficiencyLevel
    order: int;
    is_complete: bool;
    resources: List[LearningResourceDisplay] = []
    quiz: Optional[QuizDisplay] = None  # Add this line

    class Config: from_attributes = True

class TopicDisplay(BaseModel):
    id: int;
    topic_name: str;
    percentage_complete: int;
    last_accessed: datetime
    sub_topics: List[SubTopicDisplay] = []

    class Config: from_attributes = True

# --- Helper & Dependency Functions ---
def verify_password(plain_password, hashed_password): return pwd_context.verify(plain_password[:MAX_BCRYPT_LEN], hashed_password)


def get_password_hash(password): return pwd_context.hash(password[:MAX_BCRYPT_LEN])


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session: yield session


async def get_current_user(token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)) -> User:
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                          detail="Could not validate credentials",
                                          headers={"WWW-Authenticate": "Bearer"}, )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None: raise credentials_exception
    except JWTError:
        raise credentials_exception
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalars().first()
    if user is None: raise credentials_exception
    return user


# --- Gemini API Call Logic ---
def parse_gemini_json_response(response_text: str) -> dict:
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)
    json_str = match.group(1) if match else response_text
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse learning plan from AI response.")


# NEW: Prompt updated to match the desired output schema
async def generate_learning_plan(topic_data: TopicCreate, user: User) -> dict:
    model = genai.GenerativeModel('gemini-2.5-pro')
    age = topic_data.age if topic_data.age is not None else (datetime.now().year - user.birth_year)
    education = topic_data.education or user.school_year
    major = topic_data.major or user.major
    topic = topic_data.topic_name

    prompt = f"""
        As an expert curriculum designer, create a personalized learning plan for the user and topic below.
        User Profile: Age {age}, Education Level: {education}, Major/Field: {major}.
        Topic: "{topic}"

        Break the topic into logical sub-topics.
        For each sub-topic, provide:
        1. A 'title' for the sub-topic.
        2. An 'estimatedHours' string (e.g., "3 hours").
        3. A 'resources' list of 4-5 high-quality, publicly accessible learning resources. For each resource, specify its 'type', 'title', and a valid 'url'.

        The entire response must be a single JSON object with a root key "learningPlan", which is a list of the sub-topic objects.
        Example format: {{ "learningPlan": [ {{ "title": "...", "estimatedHours": "...", "resources": [{{ "type": "...", "title": "...", "url": "..." }}] }} ] }}
    """
    try:
        response = await model.generate_content_async(prompt)
        return parse_gemini_json_response(response.text)
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        raise HTTPException(status_code=503, detail="The AI service is currently unavailable.")


async def generate_quiz_from_ai(sub_topic: SubTopic) -> dict:
    model = genai.GenerativeModel('gemini-pro')

    # Format the resources into a simple list for the prompt
    resource_list = "\n".join([f"- {res.title} ({res.type}): {res.url}" for res in sub_topic.resources])

    prompt = f"""
        You are an AI assessment generator. Based on the sub-topic "{sub_topic.sub_topic_name}", and the given reading materials and resources:
        {resource_list}

        Create a multiple-choice quiz with 5 questions. The options should be concise.
        Format your response as a single JSON object with a root key "quiz" which is a list of question objects.
        For each question object, provide:
        1. 'questionText' (string)
        2. 'options' (an array of 4 strings)
        3. 'correctOptionIndex' (a number from 0 to 3, representing the index of the correct option in the array)
    """
    try:
        response = await model.generate_content_async(prompt)
        return parse_gemini_json_response(response.text)
    except Exception as e:
        print(f"Error calling Gemini API for quiz: {e}")
        raise HTTPException(status_code=503, detail="The AI service is currently unavailable for quiz generation.")


# --- API Endpoints ---
@app.post("/signup", status_code=status.HTTP_201_CREATED, tags=["Authentication"])
async def signup(user: UserSignUp, db: AsyncSession = Depends(get_db)):
    if user.password != user.confirm_password: raise HTTPException(status_code=400, detail="Passwords do not match.")
    result = await db.execute(select(User).where(User.email == user.email))
    if result.scalars().first(): raise HTTPException(status_code=400, detail="Email already registered.")
    hashed_password = get_password_hash(user.password)
    new_user = User(name=user.name, email=user.email, hashed_password=hashed_password, birth_year=user.birth_year,
                    school_year=user.school_year, major=user.major)
    db.add(new_user);
    await db.commit()
    return {"message": f"User '{new_user.name}' created successfully."}


@app.post("/login", response_model=Token, tags=["Authentication"])
async def login(form_data: UserLogin, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).where(User.email == form_data.email))
    user = result.scalars().first()
    if not (user and verify_password(form_data.password, user.hashed_password)):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}


# NEW: The response_model is removed, and the function now returns the raw learning_plan dictionary
@app.post("/topics/", response_model=None, status_code=status.HTTP_201_CREATED, tags=["Topics"])
async def create_topic(topic_data: TopicCreate, db: AsyncSession = Depends(get_db),
                       current_user: User = Depends(get_current_user)) -> Dict[str, Any]:
    # 1. Generate learning plan from Gemini
    learning_plan = await generate_learning_plan(topic_data, current_user)

    # 2. Create and save DB objects from the plan
    new_topic = Topic(topic_name=topic_data.topic_name, user_id=current_user.id)

    # 3. Parse the new JSON structure
    sub_topics_data = learning_plan.get("learningPlan", [])
    for i, sub_topic_item in enumerate(sub_topics_data):
        new_sub_topic = SubTopic(
            sub_topic_name=sub_topic_item.get("title"),
            duration=sub_topic_item.get("estimatedHours"),
            order=i + 1,
            topic=new_topic
        )
        resources_data = sub_topic_item.get("resources", [])
        for resource_item in resources_data:
            LearningResource(
                type=resource_item.get("type"),
                title=resource_item.get("title"),
                url=resource_item.get("url"),
                sub_topic=new_sub_topic
            )

    db.add(new_topic)
    await db.commit()

    # 4. Return the original learning plan JSON
    return learning_plan


@app.get("/topics/", response_model=List[TopicDisplay], tags=["Topics"])
async def get_topics_list(db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)):
    query = (select(Topic).where(Topic.user_id == current_user.id).options(
        selectinload(Topic.sub_topics).selectinload(SubTopic.resources)).order_by(Topic.created_at.desc()))
    result = await db.execute(query)
    return result.scalars().all()


@app.get("/topics/{topic_id}", response_model=TopicDisplay, tags=["Topics"])
async def get_topic_detail(topic_id: int, db: AsyncSession = Depends(get_db),
                           current_user: User = Depends(get_current_user)):
    query = (select(Topic).where(Topic.id == topic_id).where(Topic.user_id == current_user.id).options(
        selectinload(Topic.sub_topics).selectinload(SubTopic.resources)))
    result = await db.execute(query)
    topic = result.scalars().first()
    if not topic: raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Topic not found.")
    return topic


@app.patch("/subtopics/{sub_topic_id}/toggle", response_model=SubTopicDisplay, tags=["Sub-Topics"])
async def toggle_sub_topic_completion(sub_topic_id: int, db: AsyncSession = Depends(get_db),
                                      current_user: User = Depends(get_current_user)):
    """
    Toggles the is_complete status of a specific sub-topic.
    """
    # --- THIS IS THE CORRECTED QUERY ---
    query = (
        select(SubTopic)
        .where(SubTopic.id == sub_topic_id)
        .join(Topic)
        .where(Topic.user_id == current_user.id)
        .options(selectinload(SubTopic.topic))  # Eagerly load the parent Topic
    )
    result = await db.execute(query)
    sub_topic = result.scalars().first()

    if not sub_topic:
        raise HTTPException(status_code=status.HTTP_4_NOT_FOUND,
                            detail="Sub-topic not found or you don't have permission to edit it.")

    # Now, accessing sub_topic.topic will not trigger a new database call
    sub_topic.is_complete = not sub_topic.is_complete
    sub_topic.topic.last_accessed = datetime.utcnow()

    await db.commit()
    await db.refresh(sub_topic)

    # Eagerly load resources for the response model to prevent another lazy-load error
    result = await db.execute(
        select(SubTopic)
        .where(SubTopic.id == sub_topic.id)
        .options(selectinload(SubTopic.resources))
    )
    return result.scalars().first()

@app.post("/subtopics/{sub_topic_id}/generate-quiz", response_model=QuizDisplay, tags=["Sub-Topics"])
async def create_quiz_for_sub_topic(sub_topic_id: int, db: AsyncSession = Depends(get_db),
                                    current_user: User = Depends(get_current_user)):
    # 1. Fetch the sub-topic and verify ownership, pre-loading resources
    query = (
        select(SubTopic)
        .join(Topic).where(Topic.user_id == current_user.id)
        .where(SubTopic.id == sub_topic_id)
        .options(selectinload(SubTopic.resources), selectinload(SubTopic.quiz))
    )
    result = await db.execute(query)
    sub_topic = result.scalars().first()
    if not sub_topic:
        raise HTTPException(status_code=404, detail="Sub-topic not found.")

    # 2. If a quiz already exists, return it instead of creating a new one
    if sub_topic.quiz:
        # Eagerly load questions and options for the response
        result = await db.execute(select(Quiz).where(Quiz.id == sub_topic.quiz.id).options(
            selectinload(Quiz.questions).selectinload(Question.options)))
        return result.scalars().first()

    # 3. Generate the quiz content from the AI
    quiz_data = await generate_quiz_from_ai(sub_topic)

    # 4. Create database objects
    new_quiz = Quiz(sub_topic=sub_topic)
    for q_item in quiz_data.get("quiz", []):
        new_question = Question(
            question_text=q_item.get("questionText"),
            correct_option_index=q_item.get("correctOptionIndex"),
            quiz=new_quiz
        )
        for o_text in q_item.get("options", []):
            Option(option_text=o_text, question=new_question)

    db.add(new_quiz)
    await db.commit()
    await db.refresh(new_quiz)

    # 5. Eagerly load the full quiz for the response
    result = await db.execute(
        select(Quiz).where(Quiz.id == new_quiz.id).options(selectinload(Quiz.questions).selectinload(Question.options)))
    return result.scalars().first()