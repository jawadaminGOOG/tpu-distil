"""BIRD text-to-SQL; the implementation lives in `tasks/bird.py`.

`import bird` returns the `tasks.bird` module itself (not a copy of its names), so code that reads
or patches module attributes such as `bird.MAX_ROWS` acts on the module the pipeline uses.
"""
import sys

from tasks import bird as _bird

sys.modules[__name__] = _bird
