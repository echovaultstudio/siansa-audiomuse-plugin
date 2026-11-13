"""The way back in from Docker when the page's password is forgotten (0.42.3, the owner, 13 Nov):

    docker exec siansa-beats python -m beats.reset

takes the page's password and the app's key away, at once and without a restart (the service reads them from the data
folder on every request): the page opens to anyone on the network again, and new ones are set under Setup. Nothing else
is touched. ``BEATS_RESET_ACCESS=yes`` on the container does the same at its next start.
"""

import json
import os
import sqlite3
import sys
import time


def reset(data_dir):
    """Takes the password and the key away in ``data_dir``'s database. What was there: ``{"password", "api_key"}``
    (booleans). Raises ``FileNotFoundError`` when there's no database there."""
    path = os.path.join(data_dir, "beats.db")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    db = sqlite3.connect(path, timeout=30)
    try:
        row = db.execute("SELECT value FROM settings WHERE key='access'").fetchone()
        access = json.loads(row[0]) if row else {}
        had = {"password": bool(access.get("password")), "api_key": bool(access.get("api_key"))}
        with db:
            db.execute("INSERT OR REPLACE INTO settings VALUES ('access', '{}')")
            db.execute("DELETE FROM settings WHERE key='sessions'")  # anyone signed in is signed out
            if had["password"] or had["api_key"]:
                db.execute("INSERT INTO events VALUES (?,?,?)", (time.time(), "warn",
                           "The page's password and the app's key were taken away from Docker (python -m beats.reset): "
                           "set new ones under Setup"))
        return had
    finally:
        db.close()


def main(argv=None, env=None, out=sys.stdout):
    env = os.environ if env is None else env
    data_dir = env.get("DATA_DIR", "/data")
    try:
        had = reset(data_dir)
    except FileNotFoundError as e:
        print(f"No Siansa Beats database at {e}: is DATA_DIR right?", file=out)
        return 1
    except sqlite3.Error as e:
        print(f"Couldn't change the database: {e}", file=out)
        return 1
    if not had["password"] and not had["api_key"]:
        print("No password or key was set: the page is open already.", file=out)
    else:
        gone = " and ".join(name for name, was in (("the page's password", had["password"]), ("the app's key", had["api_key"])) if was)
        print(f"Taken away: {gone}. The page is open to your network now: set a new password under Setup, "
              "and give the Siansa app its new key.", file=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
