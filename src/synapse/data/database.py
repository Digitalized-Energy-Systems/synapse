"""
Module for accessing database related methods regarding protocoling results of the scenarios implemented in DissComit.
"""

import sqlite3 as sl
import os.path
import os
from pathlib import Path

CONFLICT_DB_FILE_NAME = 'data/conflict/conflict.db'

CREATE_SQL_CONF_GEN_EX = """
        CREATE TABLE IF NOT EXISTS CONFLICT_GEN_EXECUTION (
            id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
            time DATETIME UNIQUE NOT NULL,
            heat BOOLEAN NOT NULL,
            load_type TEXT NOT NULL,
            net_type TEXT NOT NULL,
            solution_index INTEGER NOT NULL,
            fitness TEXT NOT NULL
        );
"""

WRITE_SQL_CONF_GEN_EX = 'INSERT INTO CONFLICT_GEN_EXECUTION (time, heat, load_type, net_type, solution_index, fitness) values(?, ?, ?, ?, ?, ?)'
READ_SQL_CONF_GEN_EX_BY_ID = "SELECT * FROM CONFLICT_GEN_EXECUTION WHERE id = :id"
READ_SQL_CONF_GEN_EX_BY_TIME = "SELECT * FROM CONFLICT_GEN_EXECUTION WHERE time = :time"


def _init_conflict_database(con):
    con.execute(CREATE_SQL_CONF_GEN_EX)

def _ensure_path(path):
    directory = os.path.dirname(path)
    Path(directory).mkdir(parents=True, exist_ok=True)
    return path

def write_conflict_gen_execution(sys_time, heat, load_type, net_type, solution_index, fitness):
    """Insert a row in CONFLICT_GEN_EXECUTION table as part of the conflict.db.

    :param sys_time: system time
    :type sys_time: int
    :param heat: true if heat network
    :type heat: boolean
    :param load_type: type of the load profiles
    :type load_type: str
    :param net_type: type of the network
    :type net_type: str
    :param solution_index: index of the solution
    :type solution_index: int
    :param fitness: fitness value
    :type fitness: int
    :return: sql code
    :rtype: int
    """
    connection = sl.connect(_ensure_path(CONFLICT_DB_FILE_NAME))

    with connection:
        _init_conflict_database(connection)
        return connection.execute(WRITE_SQL_CONF_GEN_EX, [sys_time, heat, load_type, net_type, solution_index, fitness])

def _read_single_row(cursor):
        result_set = cursor.fetchall()
        if len(result_set) == 0:
            return None
        else:
            return result_set[0]

def read_conflict_gen_execution_by_id(id):
    """Read the row identified by the given id.

    :param id: the id
    :type id: int
    :return: row as tuple
    :rtype: tuple
    """
    connection = sl.connect(_ensure_path(CONFLICT_DB_FILE_NAME))

    with connection:
        return _read_single_row(connection.execute(READ_SQL_CONF_GEN_EX_BY_ID, {'id': id}))
        

def read_conflict_gen_execution_by_time(time):
    """Read the row identified by the given time.

    :param time: the time
    :type time: int
    :return: row as tuple
    :rtype: tuple
    """
    connection = sl.connect(_ensure_path(CONFLICT_DB_FILE_NAME))

    with connection:
        return _read_single_row(connection.execute(READ_SQL_CONF_GEN_EX_BY_TIME, {'time': time}))

def delete_conflict_database():
    """Delete the conflict.db database
    """
    if os.path.exists(CONFLICT_DB_FILE_NAME):
        os.remove(CONFLICT_DB_FILE_NAME)