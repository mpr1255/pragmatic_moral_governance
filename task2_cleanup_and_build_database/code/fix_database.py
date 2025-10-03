#%%
import sqlite3
import pandas as pd

DB_PATH = "/Volumes/8tb/data/wenming/wenming.db"

#%%
def fix_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Check if the table is already fixed
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='file_index_regular'")
    if cursor.fetchone():
        print("Database already fixed. No action needed.")
        conn.close()
        return

    # Create a new regular table with the additional column
    cursor.execute("""
    CREATE TABLE file_index_regular (
        id TEXT,
        title TEXT,
        office TEXT,
        publish TEXT,
        expiry TEXT,
        type TEXT,
        status TEXT,
        url TEXT,
        content TEXT,
        translated_content TEXT
    )
    """)

    # Copy data from the virtual table to the regular table
    cursor.execute("INSERT INTO file_index_regular (id, title, office, publish, expiry, type, status, url, content) SELECT * FROM file_index")

    # Drop the old virtual table
    cursor.execute("DROP TABLE file_index")

    # Create a new virtual table with the updated structure
    cursor.execute("""
    CREATE VIRTUAL TABLE file_index USING fts5(
        id,
        title,
        office,
        publish,
        expiry,
        type,
        status,
        url,
        content,
        translated_content
    )
    """)

    # Copy data from the regular table to the new virtual table
    cursor.execute("INSERT INTO file_index SELECT * FROM file_index_regular")

    # Drop the temporary regular table
    cursor.execute("DROP TABLE file_index_regular")

    conn.commit()
    conn.close()

    print("Database structure updated successfully.")

#%%
if __name__ == "__main__":
    fix_database()