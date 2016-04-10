from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.engine.url import URL

import settings

DeclarativeBase = declarative_base()

def db_connect():
    """
    Performs database connection using database settings from settings.py.
    Returns sqlalchemy engine instance
    """
    return create_engine(URL(**settings.DATABASE))

def create_website_table(engine):
    """"""
    DeclarativeBase.metadata.create_all(engine)

class Websites(DeclarativeBase):
    """Sqlalchemy websites model"""
    __tablename__ = "websites"

    id = Column(Integer, primary_key=True)
    link = Column('link', String, nullable=True)
    male_ratio = Column('male_ratio', Float, nullable=True)
    female_ratio = Column('female_ratio', Float, nullable=True)
