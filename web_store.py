"""
Where web_play.py keeps its games and rate-limit counters (web_app_spec.md, "Deployment").

MemoryStore (default): one process; games are lost on restart. RedisStore (web_play.py --redis-url):
games, per-game locks and rate limits live in Redis, so several server instances can serve the same
games and instances can come and go (serverless hosting, scale to zero).

Both keep games serialized (Game.to_dict / Game.from_dict in web_play.py), so the in-memory store
exercises the same code path as Redis. Interface:
  create(game_id, data, ttl)   store a new game; False if the store is full
  load(game_id) -> dict|None   a game's data
  save(game_id, data, ttl)     update a game (ttl = idle timeout: seconds until it expires)
  locked(game_id)              context manager: one request at a time per game, across instances
  hit(key, per_minute) -> bool count a request; False when over the per-minute limit
"""

import collections
import contextlib
import json
import secrets
import threading
import time


class StoreBusy(Exception):
    """A game's lock could not be taken in time (another request on it is still running)."""


class MemoryStore:
    def __init__(self, max_games=200):
        self.max_games = max_games
        self.games = {}                       # id -> (expires at, json)
        self.locks = collections.defaultdict(threading.Lock)
        self.guard = threading.Lock()
        self.rates = collections.defaultdict(collections.deque)

    def _expire(self, now):
        for gid in [g for g, (t, _) in self.games.items() if t < now]:
            del self.games[gid]
            self.locks.pop(gid, None)

    def create(self, game_id, data, ttl):
        now = time.time()
        with self.guard:
            self._expire(now)
            if len(self.games) >= self.max_games:
                return False
            self.games[game_id] = (now + ttl, json.dumps(data))
        return True

    def load(self, game_id):
        now = time.time()
        with self.guard:
            entry = self.games.get(game_id)
            if entry is None or entry[0] < now:
                return None
            return json.loads(entry[1])

    def save(self, game_id, data, ttl):
        with self.guard:
            self.games[game_id] = (time.time() + ttl, json.dumps(data))

    @contextlib.contextmanager
    def locked(self, game_id, wait=60):
        with self.guard:
            lock = self.locks[game_id]
        if not lock.acquire(timeout=wait):
            raise StoreBusy(game_id)
        try:
            yield
        finally:
            lock.release()

    def hit(self, key, per_minute):
        now = time.time()
        with self.guard:
            q = self.rates[key]
            while q and q[0] < now - 60:
                q.popleft()
            if len(q) >= per_minute:
                return False
            q.append(now)
            if len(self.rates) > 10000:           # forget idle clients
                for k in [k for k, v in self.rates.items() if not v or v[-1] < now - 60]:
                    del self.rates[k]
            return True

    def count(self):
        with self.guard:
            self._expire(time.time())
            return len(self.games)


class RedisStore:
    """Games as JSON strings under reversi:game:<id> with the idle timeout as their expiry; locks as
    reversi:lock:<id> (SET NX with an expiry, so a crashed instance can't hold one forever; released
    only by its owner, checked in a WATCH/MULTI transaction); rate limits as per-minute counters."""

    PREFIX = "reversi:"

    def __init__(self, url=None, client=None, lock_seconds=120):
        if client is None:
            import redis
            client = redis.Redis.from_url(url)
        self.r, self.lock_seconds = client, lock_seconds

    def create(self, game_id, data, ttl):
        return bool(self.r.set(self.PREFIX + "game:" + game_id, json.dumps(data), nx=True, ex=max(1, int(ttl))))

    def load(self, game_id):
        raw = self.r.get(self.PREFIX + "game:" + game_id)
        return None if raw is None else json.loads(raw)

    def save(self, game_id, data, ttl):
        self.r.set(self.PREFIX + "game:" + game_id, json.dumps(data), ex=max(1, int(ttl)))

    @contextlib.contextmanager
    def locked(self, game_id, wait=60):
        key, token = self.PREFIX + "lock:" + game_id, secrets.token_hex(8)
        deadline = time.time() + wait
        while not self.r.set(key, token, nx=True, ex=self.lock_seconds):
            if time.time() > deadline:
                raise StoreBusy(game_id)
            time.sleep(0.05)
        try:
            yield
        finally:
            self._release(key, token)

    def _release(self, key, token):
        with self.r.pipeline() as pipe:
            try:
                pipe.watch(key)
                if pipe.get(key) == token.encode():
                    pipe.multi()
                    pipe.delete(key)
                    pipe.execute()
                else:
                    pipe.unwatch()
            except Exception:                     # changed meanwhile (expired and retaken): not ours
                pass

    def hit(self, key, per_minute):
        window = int(time.time() // 60)
        k = f"{self.PREFIX}rate:{key}:{window}"
        with self.r.pipeline() as pipe:
            pipe.incr(k)
            pipe.expire(k, 120)
            count = pipe.execute()[0]
        return count <= per_minute

    def count(self):
        return sum(1 for _ in self.r.scan_iter(self.PREFIX + "game:*"))
