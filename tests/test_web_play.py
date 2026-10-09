#!/usr/bin/env python3
"""
Tests for web_play.py's API (Flask test client, tiny untrained network): games by id, input
validation, undo/redo, limits, security headers, game records, and errors that hide internals.
"""

import glob
import json
import os
import random
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import web_play as wp
from web_store import MemoryStore, RedisStore
from util.reversi import ReversiEnvCNN
from util.search import legal_moves, play_move


class TestWebPlay(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        import torch
        from util.util import get_model
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as d:
            wp.Config.model = get_model(os.path.join(d, "m"), ReversiEnvCNN(opponent="Random"), model_type="resnet",
                                        channels=8, blocks=1)
        wp.Config.model_name = "test"

    def make_store(self):
        return MemoryStore(max_games=200)

    def setUp(self):
        wp.STORE = self.make_store()
        wp.PLAYERS.clear()
        wp.Config.default_level, wp.Config.record = 3, False
        wp.Config.idle_timeout, wp.Config.rate_per_minute = 7200, 100000
        wp.Config.new_games_per_minute = 100000
        self.c = wp.app.test_client()

    def stored(self, gid):
        return wp.Game.from_dict(wp.STORE.load(gid))

    def new(self, **kw):
        r = self.c.post('/api/new_game', json={'model_color': 'white', **kw})
        self.assertEqual(r.status_code, 200, r.get_json())
        return r.get_json()

    def move(self, gid, action):
        return self.c.post('/api/make_move', json={'game_id': gid, 'action': action})

    def test_new_game_ids_and_separate_games(self):
        a, b = self.new(), self.new()
        self.assertNotEqual(a['game_id'], b['game_id'])
        self.assertRegex(a['game_id'], wp.GAME_ID_RE)
        r = self.move(a['game_id'], a['valid_moves'][0]['action']).get_json()
        self.assertEqual(r['piece_count']['black'] + r['piece_count']['white'], 6)     # move + reply
        same = self.c.get('/api/game_state', query_string={'game_id': b['game_id']}).get_json()
        self.assertEqual(same['board'], b['board'])                                    # b untouched

    def test_model_moves_first_as_black(self):
        r = self.new(model_color='black')
        self.assertEqual(r['last_move']['player'], 'model')
        self.assertEqual(r['current_player'], 'white')
        self.assertTrue(all('probability' in m for m in r['valid_moves']))

    def test_rejects_bad_moves(self):
        g = self.new()
        gid, legal = g['game_id'], {m['action'] for m in g['valid_moves']}
        illegal = next(a for a in range(64) if a not in legal)
        for action in (illegal, -1, 64, '19', 19.0, True, None, [19]):
            self.assertEqual(self.move(gid, action).status_code, 400, action)
        self.assertEqual(self.move('A' * 24, 19).status_code, 404)                    # unknown game
        self.assertEqual(self.move('../../etc', 19).status_code, 400)                 # malformed id
        self.assertEqual(self.c.post('/api/make_move', data='x', content_type='text/plain').status_code, 400)
        self.assertEqual(self.c.post('/api/new_game', json={'model_color': '../../x'}).status_code, 400)
        self.assertEqual(self.c.post('/api/new_game', json={'level': 11}).status_code, 400)
        self.assertEqual(self.c.post('/api/set_level', json={'game_id': gid, 'level': 0}).status_code, 400)

    def test_full_game_and_rules(self):
        random.seed(3)
        g = self.new()
        gid = g['game_id']
        while not g['game_over']:
            g = self.move(gid, random.choice(g['valid_moves'])['action']).get_json()
        b = np.array(g['board']).reshape(64)
        diff = int((b == 1).sum() - (b == -1).sum())
        self.assertEqual(g['winner'], 'black' if diff > 0 else 'white' if diff < 0 else 'draw')
        self.assertEqual(self.move(gid, 0).status_code, 400)                          # game over
        # the history replays by the rules (passes allowed)
        game = self.stored(gid)
        for prev, nxt in zip(game.history, game.history[1:] + [{'board_before': game.board}]):
            sign = 1 if prev['color'] == 'black' else -1
            self.assertIn(prev['action'], set(int(m) for m in legal_moves(prev['board_before'] * sign)))
            after = -play_move(prev['board_before'] * sign, prev['action']) * sign
            np.testing.assert_array_equal(after, nxt['board_before'])

    def test_undo_redo(self):
        g = self.new()
        gid, start = g['game_id'], g['board']
        first = g['valid_moves'][0]['action']
        after = self.move(gid, first).get_json()
        u = self.c.post('/api/undo', json={'game_id': gid}).get_json()
        self.assertEqual(u['board'], start)                       # back before the person's move
        self.assertEqual(u['current_player'], 'black')            # and it is the person's turn
        self.assertTrue(u['can_redo'])
        r = self.c.post('/api/redo', json={'game_id': gid}).get_json()
        self.assertEqual(r['board'], after['board'])              # level 3 is deterministic
        self.assertEqual(self.c.post('/api/undo', json={'game_id': gid}).status_code, 200)
        self.assertEqual(self.c.post('/api/undo', json={'game_id': gid}).status_code, 400)   # nothing left

    def test_levels(self):
        lv = self.c.get('/api/levels').get_json()
        self.assertEqual([l['level'] for l in lv['levels']], list(range(1, 11)))
        g = self.new(level=1)
        self.assertEqual(g['level'], 1)
        r = self.c.post('/api/set_level', json={'game_id': g['game_id'], 'level': 4}).get_json()
        self.assertEqual(r['level'], 4)
        m = self.move(g['game_id'], g['valid_moves'][0]['action']).get_json()
        self.assertTrue(m['last_move']['method'].startswith('level 4:'))

    def test_limits(self):
        if isinstance(wp.STORE, MemoryStore):
            wp.STORE.max_games = 2
            self.new(), self.new()
            self.assertEqual(self.c.post('/api/new_game', json={}).status_code, 503)
            wp.STORE = self.make_store()
        wp.Config.idle_timeout = 1                                 # games expire after 1 s idle
        old = self.new()
        time.sleep(1.2)
        self.assertEqual(self.c.get('/api/game_state', query_string={'game_id': old['game_id']}).status_code, 404)
        wp.Config.idle_timeout = 7200
        g = self.new()
        wp.Config.rate_per_minute = 3
        wp.STORE = self.make_store()
        wp.STORE.save(g['game_id'], self.stored_from(g), 7200)
        codes = [self.c.get('/api/game_state', query_string={'game_id': g['game_id']}).status_code for _ in range(5)]
        self.assertEqual(codes, [200, 200, 200, 429, 429])
        wp.Config.rate_per_minute = 100000
        self.assertEqual(self.c.post('/api/new_game', data='x' * 10000, content_type='application/json').status_code,
                         413)

    def test_headers_and_hidden_errors(self):
        r = self.c.get('/api/levels')
        for h in ('Content-Security-Policy', 'X-Content-Type-Options', 'X-Frame-Options'):
            self.assertIn(h, r.headers)
        g = self.new()
        original = wp.model_move
        wp.model_move = lambda game: 1 / 0
        try:
            r = self.move(g['game_id'], g['valid_moves'][0]['action'])
        finally:
            wp.model_move = original
        self.assertEqual(r.status_code, 500)
        self.assertEqual(r.get_json(), {'error': 'internal error'})               # no exception text
        self.assertEqual(self.c.get('/api/nothing').status_code, 404)

    def stored_from(self, state):
        return wp.Game(wp.WHITE if state['model_color'] == 'white' else wp.BLACK, state['level'],
                       state['game_id']).to_dict()

    def test_round_trip_and_busy(self):
        random.seed(4)
        g = self.new()
        for _ in range(6):
            g = self.move(g['game_id'], random.choice(g['valid_moves'])['action']).get_json()
        game = self.stored(g['game_id'])
        again = wp.Game.from_dict(json.loads(json.dumps(game.to_dict())))
        np.testing.assert_array_equal(game.board, again.board)
        self.assertEqual(game.state(), again.state())
        bad = game.to_dict()
        bad['moves'][2][0] = bad['moves'][1][0]                     # a move that can't be replayed
        with self.assertRaises(ValueError):
            wp.Game.from_dict(bad)
        with wp.STORE.locked(g['game_id']):                         # another request holds the game
            original = wp.STORE.locked
            wp.STORE.locked = lambda gid, wait=60: original(gid, wait=0.2)
            try:
                r = self.c.post('/api/undo', json={'game_id': g['game_id']})
            finally:
                wp.STORE.locked = original
        self.assertEqual(r.status_code, 409)

    def find_pass_setup(self):
        """A game (model plays white) with the person to move, a person's move h and a model reply m
        after which the person must pass while the model can move again."""
        rng = random.Random(7)
        for _ in range(3000):
            game = wp.Game(wp.WHITE, 3)
            for _ in range(rng.randrange(20, 56)):
                if game.over:
                    break
                game.apply(int(rng.choice(list(game.legal()))), "human" if game.to_move == wp.BLACK else "model")
            if game.over or game.to_move != wp.BLACK:
                continue
            board = game.board * wp.BLACK                           # the person's (black's) view
            for h in legal_moves(board):
                b2 = play_move(board, int(h))                         # white (model) to move
                if not len(legal_moves(b2)):
                    continue
                for m in legal_moves(b2):
                    b3 = play_move(b2, int(m))                        # black (person) to move?
                    if not len(legal_moves(b3)) and len(legal_moves(-b3)):
                        return game, int(h), int(m)
        self.skipTest("no pass position found")

    def test_model_sequence_when_person_passes(self):
        game, h, m = self.find_pass_setup()
        wp.STORE.create(game.id, game.to_dict(), 7200)
        original = wp.model_move
        first = [True]

        def stub(g):
            action = m if first[0] else int(g.legal()[0])
            first[0] = False
            return action, {"method": "stub", "analysis": []}
        wp.model_move = stub
        try:
            r = self.move(game.id, h).get_json()
        finally:
            wp.model_move = original
        seq = r['model_moves']
        self.assertGreaterEqual(len(seq), 2)
        self.assertEqual(seq[0], m)
        self.assertEqual(seq[-1], r['last_move']['action'])
        stored = self.stored(game.id)
        self.assertEqual([x['action'] for x in stored.history[-len(seq):]], seq)          # in play order
        self.assertTrue(all(x['by'] == 'model' for x in stored.history[-len(seq):]))
        u = self.c.post('/api/undo', json={'game_id': game.id}).get_json()
        self.assertEqual(u['model_moves'], [])

    def test_game_record(self):
        with tempfile.TemporaryDirectory() as d:
            wp.Config.record, games_dir, wp.GAMES_DIR = True, wp.GAMES_DIR, d
            try:
                g = self.new()
                self.move(g['game_id'], g['valid_moves'][0]['action'])
                files = glob.glob(os.path.join(d, '*.json'))
                self.assertEqual(len(files), 1)
                self.assertNotIn(g['game_id'], os.path.basename(files[0]))         # only an id prefix
                rec = json.load(open(files[0]))
                self.assertEqual([m['by'] for m in rec['moves']], ['human', 'model'])
                self.assertEqual(rec['level'], 3)
            finally:
                wp.GAMES_DIR = games_dir


class TestWebPlayRedis(TestWebPlay):
    """The same API tests with games, locks and rate limits in Redis (fakeredis, no server needed)."""

    def make_store(self):
        import fakeredis
        return RedisStore(client=fakeredis.FakeRedis())


if __name__ == "__main__":
    unittest.main()
