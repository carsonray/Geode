import sqlalchemy as sql
import numpy as np
from tabulate import tabulate

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import databases

def clean_list(nested):
    return [list(obj) for obj in nested]

print("Connecting to database...")

engine = databases.sqlalchemy_connect("testdb")

print()

with engine.connect() as conn:
    while True:
        query = input("Query: ")
        
        if query == "":
            break
        else:
            result = conn.execute(sql.text(query))

            if not result is None:
                rows = clean_list(result.all())
                headers = result.keys()
                print("\n" + tabulate(rows, headers=headers))

            print("\n" + "*"*70 + "\n")

print("\nDatabase closed.")
input()
