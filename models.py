import enum
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, Boolean, Enum, Text
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

class TimestampMixin:
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class ProficiencyLevel(str, enum.Enum):
    BEGINNER = "Beginner"
    INTERMEDIATE = "Intermediate"
    PROFICIENT = "Proficient"

class User(TimestampMixin, Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String); email = Column(String, unique=True, index=True)
    hashed_password = Column(String); birth_year = Column(Integer)
    school_year = Column(String); major = Column(String)
    topics = relationship("Topic", back_populates="user", cascade="all, delete-orphan")

class Topic(TimestampMixin, Base):
    __tablename__ = "topics"
    id = Column(Integer, primary_key=True, index=True)
    topic_name = Column(String, index=True)
    percentage_complete = Column(Integer, default=0)
    last_accessed = Column(DateTime(timezone=True), server_default=func.now())
    user_id = Column(Integer, ForeignKey("users.id"))
    user = relationship("User", back_populates="topics")
    sub_topics = relationship("SubTopic", back_populates="topic", cascade="all, delete-orphan")

class SubTopic(TimestampMixin, Base):
    __tablename__ = "sub_topics"
    id = Column(Integer, primary_key=True, index=True)
    sub_topic_name = Column(String)
    duration = Column(String)
    proficiency = Column(Enum(ProficiencyLevel), default=ProficiencyLevel.BEGINNER)
    order = Column(Integer)
    is_complete = Column(Boolean, default=False)
    topic_id = Column(Integer, ForeignKey("topics.id"))
    topic = relationship("Topic", back_populates="sub_topics")
    resources = relationship("LearningResource", back_populates="sub_topic", cascade="all, delete-orphan")
    # Add a one-to-one relationship to a Quiz
    quiz = relationship("Quiz", back_populates="sub_topic", uselist=False, cascade="all, delete-orphan")


class LearningResource(TimestampMixin, Base):
    __tablename__ = "learning_resources"
    id = Column(Integer, primary_key=True, index=True)
    type = Column(String) # e.g., "Video", "Article", "Book"
    title = Column(String)
    url = Column(String)
    sub_topic_id = Column(Integer, ForeignKey("sub_topics.id"))
    sub_topic = relationship("SubTopic", back_populates="resources")


class Quiz(TimestampMixin, Base):
    __tablename__ = "quizzes"
    id = Column(Integer, primary_key=True, index=True)
    sub_topic_id = Column(Integer, ForeignKey("sub_topics.id"))
    sub_topic = relationship("SubTopic", back_populates="quiz")
    questions = relationship("Question", back_populates="quiz", cascade="all, delete-orphan")

class Question(TimestampMixin, Base):
    __tablename__ = "questions"
    id = Column(Integer, primary_key=True, index=True)
    question_text = Column(String)
    correct_option_index = Column(Integer)
    quiz_id = Column(Integer, ForeignKey("quizzes.id"))
    quiz = relationship("Quiz", back_populates="questions")
    options = relationship("Option", back_populates="question", cascade="all, delete-orphan")

class Option(TimestampMixin, Base):
    __tablename__ = "options"
    id = Column(Integer, primary_key=True, index=True)
    option_text = Column(String)
    question_id = Column(Integer, ForeignKey("questions.id"))
    question = relationship("Question", back_populates="options")