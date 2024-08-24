import numpy as np
from tabulate import tabulate

import os
dirname = os.path.dirname(__file__)
import sys
sys.path.append(os.path.join(dirname,".."))

import databases

print("Connecting to database...")

db, cursor = databases.mysql_connect("testdb")

print()

while True:
    query = input("Query: ")
    
    if query == "":
        break
    else:

        try:
            cursor.execute(query)

            result = cursor.fetchall()

            if result:
                print("\n" + tabulate(result, headers=np.array(cursor.description)[:,0]))

            print("\n" + "*"*70 + "\n")

            db.commit()

        except:
            print("ERROR")

db.close()
print("\nDatabase closed.")
input()
