import sqlite3


def execute(path, proxy, **kwargs):
    connection = sqlite3.connect(path)

    try:
        proxy(connection, **kwargs)
    finally:
        connection.close()
