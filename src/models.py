from sqlalchemy import create_engine, Column, Float, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY
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


def create_db_tables(engine):
    """"""
    DeclarativeBase.metadata.create_all(engine)


class Websites(DeclarativeBase):
    """Sqlalchemy websites model"""
    __tablename__ = "websites"

    id = Column(Integer, primary_key=True)
    link = Column('link', String, nullable=True)
    male_ratio = Column('male_ratio', Float, nullable=True)
    female_ratio = Column('female_ratio', Float, nullable=True)


class WebsitesContent(DeclarativeBase):
    """Sqlalchemy websites model"""
    __tablename__ = "websites_content"

    id = Column(Integer, primary_key=True)
    link = Column('link', String, nullable=False)
    words = Column('words', ARRAY(String), nullable=False)
