"""How Siansa Beats keeps secrets (0.24.0, the owner, 11 Nov: connections are made on the page, and no password is
kept as plain text or in the container's YAML).

Two kinds of secret, kept two ways:

- **What Siansa Beats checks** (the page's password, the app's key) is **hashed**: only a salted scrypt hash of the
  password and a SHA-256 of the key (random, so it needs no salt or stretching) are kept. Neither can be turned back
  into the secret; a key is shown once, when it's made.
- **What Siansa Beats has to say to another server** (Navidrome's password, Emby's and Jellyfin's sign-in token,
  AudioMuse-AI's token) can't be hashed: a hash can't be sent as the password. It's **encrypted** (Fernet: AES with a
  signature) with a key that's not in the database: ``BEATS_SECRET_KEY`` when the container has one, else the file
  ``secret.key`` in the data folder, made on the first start and readable by Siansa Beats alone. A copy of the
  database without that key says nothing. Emby's and Jellyfin's passwords aren't kept at all: they're used once to
  sign in, and the token the server gives is kept.

Nothing here is ever logged, sent to the page or put in a report.
"""

import base64
import hashlib
import hmac
import os
import secrets

from cryptography.fernet import Fernet, InvalidToken

SCRYPT = {"n": 2 ** 14, "r": 8, "p": 1}  # ~50 ms: checked once per browser, then remembered while it runs
MARK = "enc:"  # before an encrypted value


class Vault:
    def __init__(self, data_dir, key=None):
        """``key``: ``BEATS_SECRET_KEY`` (any text: its SHA-256 is the key), else the data folder's ``secret.key``."""
        self.from_env = bool(key)
        if key:
            raw = base64.urlsafe_b64encode(hashlib.sha256(key.encode()).digest())
        else:
            path = os.path.join(data_dir, "secret.key")
            if not os.path.exists(path):
                os.makedirs(data_dir, exist_ok=True)
                with open(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
                    f.write(Fernet.generate_key())
            with open(path, "rb") as f:
                raw = f.read().strip()
        self._fernet = Fernet(raw)
        self._seen = set()  # passwords that were right, as this run's own HMAC: the slow hash isn't done on every request
        self._run_key = secrets.token_bytes(32)

    # ---- what's said to other servers: encrypted ----

    def seal(self, text):
        """``text`` encrypted, for the database ("" stays "")."""
        return MARK + self._fernet.encrypt(text.encode()).decode() if text else ""

    def open(self, sealed):
        """What :meth:`seal` hid; "" when there's nothing, or it was sealed with another key (the key file was lost or
        ``BEATS_SECRET_KEY`` changed: the secret has to be entered again)."""
        if not sealed or not str(sealed).startswith(MARK):
            return ""
        try:
            return self._fernet.decrypt(str(sealed)[len(MARK):].encode()).decode()
        except (InvalidToken, ValueError):
            return ""

    # ---- what Siansa Beats checks: hashed ----

    @staticmethod
    def hash_password(password):
        salt = secrets.token_bytes(16)
        digest = hashlib.scrypt(password.encode(), salt=salt, dklen=32, **SCRYPT)
        return f"scrypt${SCRYPT['n']}${SCRYPT['r']}${SCRYPT['p']}${salt.hex()}${digest.hex()}"

    def check_password(self, given, stored):
        """Whether ``given`` is the password ``stored`` is the hash of."""
        if not stored or not given:
            return False
        mark = hmac.new(self._run_key, (stored + "\0" + given).encode(), "sha256").digest()
        if mark in self._seen:
            return True
        try:
            kind, n, r, p, salt, digest = stored.split("$")
            if kind != "scrypt":
                return False
            found = hashlib.scrypt(given.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=32)
        except (ValueError, TypeError):
            return False
        if hmac.compare_digest(found, bytes.fromhex(digest)):
            if len(self._seen) > 64:
                self._seen.clear()
            self._seen.add(mark)
            return True
        return False

    @staticmethod
    def new_key():
        """A key for the app: ``(the key, shown once; its hash, kept)``."""
        key = secrets.token_urlsafe(32)
        return key, Vault.hash_key(key)

    @staticmethod
    def hash_key(key):
        return "sha256$" + hashlib.sha256(key.encode()).hexdigest()

    @staticmethod
    def check_key(given, stored):
        return bool(stored and given) and hmac.compare_digest(Vault.hash_key(given), stored)
