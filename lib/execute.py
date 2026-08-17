import sqlite3


def execute(db_path, proxy, **kwargs):
    connection = sqlite3.connect(db_path)

    try:
        proxy(connection, **kwargs)
    finally:
        connection.close()
