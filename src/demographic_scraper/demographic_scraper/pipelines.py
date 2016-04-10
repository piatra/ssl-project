from sqlalchemy.orm import sessionmaker
from models import Websites, db_connect, create_db_tables


class WebsiteDemographicPipeline(object):
    def __init__(self):
        """
        Initializes database connection and sessionmaker.
        """
        engine = db_connect()
        create_db_tables(engine)
        self.Session = sessionmaker(bind=engine)

    def process_item(self, item, spider):
        """Save websites in the database.
        This method is called for every item pipeline component.

        """
        session = self.Session()
        entry = Websites(**item)
        print "Process ITEM"
        print entry

        try:
            session.add(entry)
            session.commit()
        except:
            session.rollback()
            raise
        finally:
            session.close()

        return item
