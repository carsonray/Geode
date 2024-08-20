import mysql.connector as connector
import sqlalchemy as sql
import numpy as np

import json
# Database connections

CONFIG = "db_config.json"
CURR_CONN = "home"


# Loads configuration file
config_file = open(CONFIG)
connections = json.load(config_file)

def connect_obj(database, connection=None):
    connection = CURR_CONN if connection is None else connection

    obj = connections[connection].copy()
    obj["db"] = database

    return obj

def mysql_connect(database, connection=None):
    obj = connect_obj(database, connection)

    conn = connector.connect(**obj)
    cursor = conn.cursor()

    return (conn, cursor)

def sqlalchemy_connect(database, **kwargs):
    connection = kwargs.get("connection")
    obj = connect_obj(database, connection)
    
    connect_str = "mysql://{user}:{password}@{host}/{db}".format(**obj)
    engine = sql.create_engine(connect_str, **kwargs)

    return engine
