"""No secret as plain text (0.24.0): the connections are made on the page, the page's password and the app's key
are hashed, what has to be said to another server is encrypted, and Emby's and Jellyfin's passwords aren't kept at
all. What the container's settings said until then is taken over once."""

import base64
import glob
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from beats.server import App, Config, make_server  # noqa: E402
from beats.vault import Vault  # noqa: E402
from fakes import PASSWORD, SECOND_PASSWORD, TOKEN, USER, FakeMediaServer  # noqa: E402
from test_service import FakeStyle, FakeTracker, Service, fake_decode, fake_sound  # noqa: E402

EMBY_PASSWORD = "emby-pass-7712"


class VaultTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_what_is_sealed_opens_only_with_its_key(self):
        vault = Vault(self.tmp)
        sealed = vault.seal("sesame")
        self.assertTrue(sealed.startswith("enc:"))
        self.assertNotIn("sesame", sealed)
        self.assertNotEqual(sealed, vault.seal("sesame"))  # never the same twice
        self.assertEqual(vault.open(sealed), "sesame")
        self.assertEqual(Vault(self.tmp).open(sealed), "sesame")  # the key file is kept
        self.assertEqual(oct(os.stat(os.path.join(self.tmp, "secret.key")).st_mode & 0o777), "0o600")
        self.assertEqual((vault.seal(""), vault.open(""), vault.open(None), vault.open("sesame")), ("", "", "", ""))
        # Another key (the file lost, or another BEATS_SECRET_KEY): nothing, not an error.
        other = Vault(tempfile.mkdtemp())
        self.assertEqual(other.open(sealed), "")
        named = Vault(self.tmp, "a key from the container")
        self.assertTrue(named.from_env)
        self.assertEqual(Vault(tempfile.mkdtemp(), "a key from the container").open(named.seal("x")), "x")
        self.assertEqual(vault.open(named.seal("x")), "")

    def test_a_password_is_checked_against_its_hash(self):
        vault = Vault(self.tmp)
        stored = Vault.hash_password("correct horse")
        self.assertTrue(stored.startswith("scrypt$"))
        self.assertNotIn("correct horse", stored)
        self.assertNotEqual(stored, Vault.hash_password("correct horse"))  # salted
        self.assertTrue(vault.check_password("correct horse", stored))
        self.assertTrue(vault.check_password("correct horse", stored))  # remembered while it runs
        self.assertFalse(vault.check_password("correct horsf", stored))
        self.assertFalse(vault.check_password("", stored))
        self.assertFalse(vault.check_password("correct horse", ""))
        self.assertFalse(vault.check_password("correct horse", "correct horse"))  # a plain text isn't a hash
        # Remembering a right password doesn't let it in once the password is changed.
        self.assertFalse(vault.check_password("correct horse", Vault.hash_password("another one")))

    def test_a_key_for_the_app_is_kept_as_its_hash(self):
        key, stored = Vault.new_key()
        self.assertGreaterEqual(len(key), 40)
        self.assertNotIn(key, stored)
        self.assertTrue(Vault.check_key(key, stored))
        self.assertFalse(Vault.check_key(key + "x", stored))
        self.assertFalse(Vault.check_key("", stored))
        self.assertFalse(Vault.check_key(key, None))


class SecretsTest(Service):
    def restart(self, **env):
        """The service again on the same data folder, with only ``env`` on the container."""
        self.app.worker.stop()
        self.server.shutdown()
        self.server.server_close()
        self.app.store.close()
        self.app = App(Config({"DATA_DIR": os.path.join(self.tmp, "data"), "PORT": "0", **env}), tracker=FakeTracker(),
                       decode=fake_decode, style=FakeStyle(os.path.join(self.tmp, "models"), on=False), sound=fake_sound)
        self.server = make_server(self.app, host="127.0.0.1", port=0)
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.app.worker.start()

    def refused(self, path, method="POST", body=None, **more):
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call(path, method, body, **more)
        return e.exception.code, json.loads(e.exception.read() or b"{}").get("error", "")

    def on_disk(self):
        """Everything in the data folder but the key file, as bytes."""
        self.app.store.db.execute("PRAGMA wal_checkpoint(FULL)")
        data = b""
        for path in glob.glob(os.path.join(self.tmp, "data", "**"), recursive=True):
            if os.path.isfile(path) and not path.endswith("secret.key"):
                with open(path, "rb") as f:
                    data += f.read()
        return data

    def test_nothing_is_kept_as_plain_text_and_the_container_needs_none_of_it(self):
        self.start(password="page-pass-5531", api_key="app-key-from-yaml-8842")
        emby = FakeMediaServer(self.music).start()
        self.addCleanup(emby.stop)
        emby.password = EMBY_PASSWORD
        self.call("/api/servers", "POST", {"kind": "emby", "name": "Emby", "url": emby.url, "user": USER, "password": EMBY_PASSWORD})
        self.call("/api/servers", "POST", {"kind": "navidrome", "name": "Second", "url": self.fake.url, "user": "second",
                                           "password": SECOND_PASSWORD})
        self.call("/api/libraries", "POST", {"ids": ["1"]})
        self.call("/api/libraries?server=emby", "POST", {"ids": ["lib1"]})
        self.wait_done()
        key = self.call("/api/access/key", "POST")["key"]

        disk = self.on_disk()
        for secret in (EMBY_PASSWORD, SECOND_PASSWORD, "page-pass-5531", "app-key-from-yaml-8842", key, "tok-1"):
            self.assertNotIn(secret.encode(), disk, secret)
        self.assertIn(b"enc:", disk)
        self.assertIn(b"scrypt$", disk)
        saved = {s["id"]: s for s in self.app.store.setting("servers")}
        self.assertEqual(sorted(saved["emby"]), ["id", "kind", "login", "name", "url", "user"])  # no password at all
        self.assertEqual(sorted(saved["main"]), ["id", "kind", "name", "secret", "url", "user"])
        self.assertNotIn("password", json.dumps(self.app.store.setting("servers")))
        # Never said either: the page's answers, the report and the log.
        for path in ("/api/servers", "/api/status", "/api/debug", "/api/log?limit=5000", "/api/v1/capabilities"):
            raw, data = self.call(path, raw=True)
            for secret in (EMBY_PASSWORD, SECOND_PASSWORD, "page-pass-5531", "app-key-from-yaml-8842", key, "tok-1", "enc:", "scrypt$"):
                self.assertNotIn(secret, data.decode(), f"{secret} in {path}")
        left = self.call("/api/status")["connections"]["left_in_container"]
        self.assertEqual(left, ["NAVIDROME_PASSWORD", "AUDIOMUSE_TOKEN", "BEATS_PASSWORD", "BEATS_API_KEY"])

        # The container's settings gone from the YAML: everything still works from the data folder.
        self.restart()
        self.assertEqual(self.refused("/api/status", "GET", password="")[0], 401)
        self.assertEqual(self.refused("/api/status", "GET", password="page-pass-5532")[0], 401)
        status = self.call("/api/status")
        self.assertEqual(status["connections"]["left_in_container"], [])
        self.assertEqual([(s["id"], s["signed_in"], s["songs"]) for s in status["servers"]],
                         [("main", True, 4), ("emby", True, 4), ("navidrome", True, 0)])
        self.assertEqual(self.call("/api/libraries")[0]["name"], "Music")  # Navidrome takes the kept password
        logins = emby.logins
        self.assertEqual(self.call("/api/libraries?server=emby")[0]["id"], "lib1")  # Emby the kept token
        self.assertEqual(emby.logins, logins)  # no new sign-in: there's no password to sign in with
        self.assertEqual(self.call("/api/v1/similar/artists?artist=Ecsape")["source"], "audiomuse")
        for bearer in (key, "page-pass-5531"):
            self.assertEqual(self.call("/api/v1/capabilities", password="", headers={"Authorization": f"Bearer {bearer}"})["api"], 1)
        # The key from the YAML stopped working when a new one was made.
        self.assertEqual(self.refused("/api/v1/capabilities", "GET", password="",
                                      headers={"Authorization": "Bearer app-key-from-yaml-8842"})[0], 401)

    def test_the_containers_settings_are_taken_once(self):
        self.start(password="page-pass-5531")
        self.call("/api/access/password", "POST", {"password": "a newer password"})
        self.password = "a newer password"
        self.call("/api/servers/main", "DELETE")
        self.call("/api/audiomuse", "DELETE")
        # The YAML still says what it said: none of it comes back.
        self.restart(NAVIDROME_URL=self.fake.url, NAVIDROME_USER=USER, NAVIDROME_PASSWORD=PASSWORD,
                     AUDIOMUSE_URL=self.fake.url, AUDIOMUSE_TOKEN=TOKEN, BEATS_PASSWORD="page-pass-5531")
        self.assertEqual(self.refused("/api/status", "GET", password="page-pass-5531")[0], 401)
        status = self.call("/api/status")
        self.assertEqual(status["servers"], [])
        self.assertFalse(status["connections"]["audiomuse"])
        self.assertEqual(status["connections"]["left_in_container"], ["NAVIDROME_PASSWORD", "AUDIOMUSE_TOKEN", "BEATS_PASSWORD"])

    def test_the_pages_password_and_the_apps_key(self):
        self.start()
        self.assertEqual(self.call("/api/status")["connections"]["password"], False)
        # A key alone would shut browsers out of the page (they can only say a password): not without one.
        code, why = self.refused("/api/access/key")
        self.assertEqual(code, 400)
        self.assertIn("Set the page's password first", why)
        code, why = self.refused("/api/access/password", body={"password": "short"})
        self.assertEqual((code, why), (400, "At least 8 characters"))
        self.assertEqual(self.call("/api/access/password", "POST", {"password": "long enough"}), {"password": True})
        self.assertEqual(self.refused("/api/status", "GET")[0], 401)
        self.assertEqual(self.refused("/api/access/password", "DELETE")[0], 401)  # not without it
        self.password = "long enough"
        self.assertTrue(self.call("/api/status")["connections"]["password"])
        self.assertIn("The page's password was set", [e["message"] for e in self.call("/api/status")["events"]])
        self.assertNotIn("long enough", json.dumps(self.call("/api/status")))

        first = self.call("/api/access/key", "POST")["key"]
        second = self.call("/api/access/key", "POST")["key"]
        self.assertNotEqual(first, second)
        bearer = lambda k: {"Authorization": f"Bearer {k}"}  # noqa: E731
        self.assertEqual(self.refused("/api/v1/capabilities", "GET", password="", headers=bearer(first))[0], 401)
        self.assertEqual(self.call("/api/v1/capabilities", password="", headers=bearer(second))["api"], 1)
        self.assertEqual(self.call("/api/access/key", "DELETE"), {"api_key": False})
        self.assertEqual(self.refused("/api/v1/capabilities", "GET", password="", headers=bearer(second))[0], 401)

        third = self.call("/api/access/key", "POST")["key"]
        self.assertEqual(self.call("/api/access/password", "DELETE"), {"password": False})
        self.password = ""
        self.assertEqual(self.call("/api/health")["status"], "ok")
        cn = self.call("/api/status")["connections"]  # open again, and the key gone with the password
        self.assertEqual((cn["password"], cn["api_key"]), (False, False))
        self.assertEqual(self.call("/api/v1/capabilities", headers=bearer(third))["api"], 1)  # nothing is asked for

        # Forgotten: BEATS_RESET_ACCESS on the container is the way back in.
        self.call("/api/access/password", "POST", {"password": "forgotten soon"})
        self.restart()
        self.assertEqual(self.refused("/api/status", "GET")[0], 401)
        self.restart(BEATS_RESET_ACCESS="yes")
        self.assertFalse(self.call("/api/status")["connections"]["password"])

    def test_audiomuse_is_connected_on_the_page(self):
        self.start()
        self.call("/api/audiomuse", "DELETE")
        self.assertFalse(self.call("/api/status")["connections"]["audiomuse"])
        self.assertEqual(self.refused("/api/v1/similar/artists?artist=Ecsape", "GET")[0], 502)
        self.assertEqual(self.refused("/api/audiomuse", body={"url": "nas:8000"}), (400, "The address starts with http:// or https://"))
        code, why = self.refused("/api/audiomuse", body={"url": self.fake.url, "token": "not it"})
        self.assertEqual((code, why), (502, "AudioMuse-AI didn't take the token"))
        self.assertEqual(self.refused("/api/audiomuse", body={"url": self.fake.url})[1], "AudioMuse-AI wants a token: one of its API tokens")
        self.assertEqual(self.refused("/api/audiomuse", body={"url": "http://127.0.0.1:1", "token": TOKEN})[0], 502)
        self.assertFalse(self.call("/api/status")["connections"]["audiomuse"])  # nothing kept from the tries
        self.assertEqual(self.call("/api/audiomuse/test", "POST", {"url": self.fake.url, "token": TOKEN})["plugin"], "siansa_analysis")
        self.assertFalse(self.call("/api/status")["connections"]["audiomuse"])
        self.assertEqual(self.call("/api/audiomuse", "POST", {"url": self.fake.url + "/", "token": TOKEN}),
                         {"plugin": "siansa_analysis", "url": self.fake.url})
        cn = self.call("/api/status")["connections"]
        self.assertEqual((cn["audiomuse"], cn["audiomuse_url"], cn["audiomuse_token"]), (True, self.fake.url, True))
        self.assertEqual(self.call("/api/v1/similar/artists?artist=Ecsape")["source"], "audiomuse")
        # The same address again with no token: the kept one stays.
        self.assertEqual(self.call("/api/audiomuse", "POST", {"url": self.fake.url})["plugin"], "siansa_analysis")
        self.assertNotIn(TOKEN.encode() + b'"', self.on_disk().replace(b'"token"', b""))

    def test_a_sign_in_the_server_stops_taking_is_made_again_with_the_password(self):
        self.start()
        jellyfin = FakeMediaServer(self.music, jellyfin=True).start()
        self.addCleanup(jellyfin.stop)
        self.call("/api/servers", "POST", {"kind": "jellyfin", "name": "Den", "url": jellyfin.url, "user": USER, "password": PASSWORD})
        jellyfin.token = "tok-2"  # signed out there (its devices cleared, or the password changed)
        code, why = self.refused("/api/libraries?server=jellyfin", "GET")
        self.assertEqual(code, 502)
        self.assertIn("no longer takes the saved sign-in: enter the password again", why)
        self.assertEqual(self.refused("/api/servers/jellyfin", body={"password": "wrong"})[0], 502)
        self.assertEqual(self.call("/api/servers/jellyfin", "POST", {"password": PASSWORD})["signed_in"], True)
        self.assertEqual(self.call("/api/libraries?server=jellyfin")[0]["id"], "lib1")

    def test_a_lost_key_says_so_and_asks_again(self):
        self.start()
        os.remove(os.path.join(self.tmp, "data", "secret.key"))
        self.restart()
        status = self.call("/api/status")
        self.assertEqual([(s["id"], s["signed_in"]) for s in status["servers"]], [("main", False)])
        self.assertEqual(self.refused("/api/libraries", "GET")[0], 502)
        self.assertEqual(self.call("/api/servers/main", "POST", {"password": PASSWORD})["signed_in"], True)
        self.assertEqual(self.call("/api/libraries")[0]["name"], "Music")

    def test_a_password_kept_as_typed_by_0_23_is_encrypted(self):
        self.start()
        self.app.store.set_setting("servers", self.app.store.setting("servers") + [
            {"id": "navidrome", "kind": "navidrome", "name": "Second", "url": self.fake.url, "user": "second",
             "password": SECOND_PASSWORD}])
        self.restart()
        self.assertNotIn(SECOND_PASSWORD.encode(), self.on_disk())
        self.assertEqual(self.call("/api/libraries?server=navidrome")[0]["name"], "Music")


if __name__ == "__main__":
    unittest.main()
