"""Resumable download of every Weibo2014 subject through MOABB into raw/ (run gen_metadata.py first)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from IO.loader import fetch_moabb

fetch_moabb(os.path.dirname(os.path.abspath(__file__)))
