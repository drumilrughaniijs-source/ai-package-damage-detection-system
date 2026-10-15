"""Persistent inspection history: SQLite (Python standard library) + the annotated / original images on disk.

data/inspections.db       one row per inspection, including the full JSON report
data/images/<ID>_*.jpg    annotated result and original photo (file names come from the inspection ID only,
                          never from user input)
"""
import json
import os
import sqlite3
from contextlib import contextmanager

import cv2
import pandas as pd

import inspection as ins

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, 'data')
IMG_DIR = os.path.join(DATA_DIR, 'images')
DB_PATH = os.path.join(DATA_DIR, 'inspections.db')
MAX_RECORDS = 2000                                    # oldest records (and their images) are pruned beyond this

COLUMNS = ['id', 'timestamp', 'filename', 'source', 'engine', 'decision', 'damage_type', 'severity',
           'p_intact', 'top_class', 'top_class_probability', 'threshold', 'damaged_area_pct', 'processing_ms']


def _connect():
    os.makedirs(IMG_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH, timeout=10)
    con.execute('''CREATE TABLE IF NOT EXISTS inspections (
        id TEXT PRIMARY KEY, timestamp TEXT, filename TEXT, source TEXT, engine TEXT, decision TEXT,
        damage_type TEXT, severity TEXT, p_intact REAL, top_class TEXT, top_class_probability REAL, threshold REAL,
        damaged_area_pct REAL, processing_ms REAL, report_json TEXT)''')
    return con


@contextmanager
def _db():
    """Connection that commits on success, rolls back on error and is always closed."""
    con = _connect()
    try:
        with con:
            yield con
    finally:
        con.close()


def _img_path(rid, kind):
    return os.path.join(IMG_DIR, f'{rid}_{kind}.jpg')


def save(report):
    """Store one inspection report and its images. Errors are returned (not raised) so a full disk never
    breaks an inspection."""
    try:
        cv2.imwrite(_img_path(report['id'], 'annotated'), report['annotated'], [cv2.IMWRITE_JPEG_QUALITY, 90])
        orig = report['_analysis']['image']
        cv2.imwrite(_img_path(report['id'], 'original'), orig, [cv2.IMWRITE_JPEG_QUALITY, 90])
        row = [report.get(c) for c in COLUMNS] + [ins.report_json(report)]
        with _db() as con:
            con.execute(f'INSERT OR REPLACE INTO inspections ({", ".join(COLUMNS)}, report_json) '
                        f'VALUES ({", ".join("?" * (len(COLUMNS) + 1))})', row)
            _prune(con)
        return None
    except (OSError, sqlite3.Error, cv2.error) as e:
        return f'The inspection could not be saved to history ({e}).'


def _prune(con):
    old = [r[0] for r in con.execute('SELECT id FROM inspections ORDER BY timestamp DESC LIMIT -1 OFFSET ?',
                                     (MAX_RECORDS,))]
    for rid in old:
        _delete_files(rid)
    con.executemany('DELETE FROM inspections WHERE id = ?', [(r,) for r in old])


def _delete_files(rid):
    for kind in ('annotated', 'original'):
        try:
            os.remove(_img_path(rid, kind))
        except FileNotFoundError:
            pass


def load(limit=MAX_RECORDS):
    """All stored inspections, newest first, as a DataFrame."""
    with _db() as con:
        return pd.read_sql_query(f'SELECT {", ".join(COLUMNS)} FROM inspections ORDER BY timestamp DESC, rowid DESC '
                                 'LIMIT ?', con, params=(limit,))


def get(rid):
    """(report dict, annotated BGR, original BGR) for one inspection, or None."""
    with _db() as con:
        row = con.execute('SELECT report_json FROM inspections WHERE id = ?', (rid,)).fetchone()
    if row is None:
        return None
    return json.loads(row[0]), cv2.imread(_img_path(rid, 'annotated')), cv2.imread(_img_path(rid, 'original'))


def clear():
    with _db() as con:
        for (rid,) in con.execute('SELECT id FROM inspections').fetchall():
            _delete_files(rid)
        con.execute('DELETE FROM inspections')
