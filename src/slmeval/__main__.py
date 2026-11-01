"""``python -m slmeval``, for when the console script is not on PATH.

A fresh virtualenv on Windows often is not activated in the shell the user
happens to be in, and "slmeval: command not found" five seconds after a
successful install is a bad first five minutes. This is the same entry point
the ``slmeval`` command uses.
"""

from __future__ import annotations

from slmeval.cli.main import main

if __name__ == "__main__":
    main()
