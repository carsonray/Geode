import mysql.connector as connector
import sqlalchemy as sql
import numpy as np

def connect_obj(db_config, database):
    obj = db_config.copy()
    obj["db"] = database

    return obj

def mysql_connect(db_config, database):
    obj = connect_obj(db_config, database)

    conn = connector.connect(**obj)
    cursor = conn.cursor()

    return (conn, cursor)

def sqlalchemy_connect(db_config, database, **kwargs):
    obj = connect_obj(db_config, database)
    
    connect_str = "mysql+pymysql://{user}:{password}@{host}/{db}".format(**obj)
    engine = sql.create_engine(connect_str, **kwargs)

    return engine
