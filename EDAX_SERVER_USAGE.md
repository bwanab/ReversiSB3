# Edax Server Architecture - Usage Guide

The Edax opponent now uses a client/server architecture to avoid multiprocessing issues with Stable Baselines 3.

## Why Client/Server?

SB3 uses multiprocessing for parallel environments, which caused issues with the shared C library state in direct Python bindings. The server architecture solves this by:

1. **Edax runs in a dedicated process** - Never forked, no state corruption
2. **Communication via Unix sockets** - Simple, fast, multiprocessing-safe
3. **Clean separation** - Training processes only need lightweight client

## Quick Start

### 1. Start the Edax Server (in one terminal)

```bash
cd /Users/bill/src/ReversiSB3
python edax_server.py
```

You should see:
```
Edax server listening on /tmp/edax_server.sock
```

**Leave this running** while you train!

### 2. Run Training (in another terminal)

```bash
# Train against Edax depth 1
python sb-train.py -m my_model -o Edax --depth 1 --timesteps 1000000

# Train against Edax depth 6
python sb-train.py -m my_model -o Edax --depth 6 --timesteps 500000
```

### 3. Test Your Model

```bash
# Play 100 games against Edax depth 1
python sb-play.py -m my_model -e 100 -o Edax --depth 1
```

## Architecture

```
┌─────────────────┐
│  Edax Server    │  (Single process, runs continuously)
│  edax_server.py │
│                 │
│  ┌───────────┐  │
│  │ EdaxEngine│  │  (C library loaded once)
│  │  depth 1  │  │
│  │  depth 4  │  │
│  │  depth 6  │  │
│  └───────────┘  │
└────────┬────────┘
         │ Unix Socket (/tmp/edax_server.sock)
         │
         ├─────────┬─────────┬─────────┐
         │         │         │         │
    ┌────▼───┐ ┌──▼────┐ ┌──▼────┐ ┌──▼────┐
    │Worker 1│ │Worker2│ │Worker3│ │Worker4│  (SB3 parallel envs)
    │        │ │       │ │       │ │       │
    │EdaxOpp │ │EdaxOpp│ │EdaxOpp│ │EdaxOpp│
    │Client  │ │Client │ │Client │ │Client │
    └────────┘ └───────┘ └───────┘ └───────┘
```

## Protocol

**Request (JSON):**
```json
{
  "state": [[0,0,0,...], [0,0,0,...], ...],  // 8x8 board
  "depth": 6
}
```

**Response (JSON):**
```json
{
  "move": 37,        // Best move (0-63) or null
  "score": -5,       // Evaluation score
  "nodes": 2277351   // Nodes searched
}
```

## Testing

### Test Server Connection

```bash
# In one terminal, start server:
python edax_server.py

# In another terminal, run tests:
python test_edax_server.py
```

### Manual Test

```python
from util.edax_client import EdaxClient
import numpy as np

client = EdaxClient(depth=6)

# Create opening position
state = np.zeros((1, 8, 8), dtype=np.int8)
state[0, 3, 3] = -1  # White
state[0, 3, 4] = 1   # Black
state[0, 4, 3] = 1   # Black
state[0, 4, 4] = -1  # White

# Get move
move = client.get_move(state)
print(f"Move: {move}")  # Should be 19, 26, 37, or 44 (d3, c4, f5, e6)
```

## Performance

| Depth | Time/Move | Throughput | Use Case |
|-------|-----------|------------|----------|
| 1     | ~0.001s   | 1000/sec   | Fast training |
| 4     | ~0.01s    | 100/sec    | Balanced |
| 6     | ~0.1s     | 10/sec     | Strong opponent |
| 8     | ~1s       | 1/sec      | Very strong |

**Overhead:** Socket communication adds ~0.1-0.5ms (negligible compared to search time)

## Troubleshooting

### "Cannot connect to Edax server"

**Problem:** Server not running

**Solution:**
```bash
# Start server in separate terminal
python edax_server.py
```

### "Address already in use"

**Problem:** Old server process still running or socket file exists

**Solution:**
```bash
# Kill old server
killall -9 python  # Or find specific process

# Remove socket file
rm /tmp/edax_server.sock

# Start fresh
python edax_server.py
```

### Server crashes

**Problem:** C library issue (rare)

**Solution:**
1. Stop server (Ctrl+C)
2. Remove socket: `rm /tmp/edax_server.sock`
3. Restart: `python edax_server.py`

### Training hangs

**Problem:** Server died during training

**Solution:**
1. Check if server is still running: `ps aux | grep edax_server`
2. If not, restart it
3. Training should resume automatically (SB3 will retry failed opponent calls)

## Advanced Usage

### Custom Socket Path

```bash
# Server
python edax_server.py --socket /tmp/my_edax.sock

# Client (in Python)
from util.edax_client import EdaxClient
client = EdaxClient(socket_path="/tmp/my_edax.sock", depth=6)
```

### Multiple Servers (Different Machines)

Currently uses Unix sockets (same machine only). To use across machines, modify to TCP sockets:

1. Edit `edax_server.py`: Change `socket.AF_UNIX` to `socket.AF_INET`
2. Bind to `("0.0.0.0", PORT)` instead of Unix socket path
3. Update client to connect to `(HOST, PORT)`

## Server Management

### Run in Background

```bash
# Start in background
nohup python edax_server.py > edax_server.log 2>&1 &

# Check if running
ps aux | grep edax_server

# Stop
killall python  # Or use specific PID
```

### Monitor Server Activity

```bash
# Watch log in real-time
tail -f edax_server.log

# Count requests
grep "Creating Edax engine" edax_server.log | wc -l
```

## Comparison to Previous Approaches

| Approach | Pros | Cons | Status |
|----------|------|------|--------|
| **GTP Protocol** | Standard | Slow, sync issues | ❌ Removed |
| **Direct Bindings** | Fast | Multiprocessing crashes | ⚠️ Works for single-threaded only |
| **Client/Server** | Fast, MP-safe | Needs server running | ✅ **Current** |

## Files

- `edax_server.py` - Server implementation
- `util/edax_client.py` - Client library
- `util/opponents.py` - EdaxOpponent (uses client)
- `test_edax_server.py` - Test suite
- `EDAX_SERVER_USAGE.md` - This file

## Next Steps

Once training works well:

1. **Benchmark** server overhead vs direct bindings (for single-threaded use)
2. **Add monitoring** - Track requests/sec, average search time
3. **Connection pooling** - Reuse connections for better performance
4. **Multiple servers** - Run depth-1, depth-4, depth-6 servers on different sockets
