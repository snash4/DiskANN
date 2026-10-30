//! Lean rerank fetch for the Garnet FP backend in **cluster** mode.
//!
//! A Redis-protocol cluster rejects a multi-key `MGET` whose keys span hash slots, and the L keys of
//! one rerank almost always fall in L different slots, so the fetch is L single-key `GET`s grouped
//! per owning node. redis-rs's cluster pipeline does this correctly but expensively: per key it
//! builds a command object, routes it through a hash map, and on the way back parses each reply with
//! a separate call into its general-purpose parser, allocating a `Value` per key, then converts and
//! copies again. Measured on a 3-node Garnet cluster (100 keys, dim 100, service stopped), that path
//! takes ~0.25–0.27 ms per fetch vs ~0.13–0.17 ms for a lean reader, at about half the throughput —
//! and inside a CPU-saturated search service the extra client work is amplified several times.
//!
//! This module is the lean path, used **only for the rerank fetch**:
//! - a slot → node table, built from `CLUSTER SLOTS` at startup and refreshed after any error;
//! - per pooled entry, one plain TCP connection per node and reusable buffers;
//! - per fetch: compute each key's slot (CRC16), append its `GET` to its node's buffer, write every
//!   node's buffer, then read each node's replies with large reads and parse them in one pass;
//!   values are handed to the caller's decoder straight from the read buffer (one copy).
//!
//! Anything unusual — a redirection (`MOVED`/`ASK`), an error reply, a timeout, an I/O error, a
//! slot not covered by the table — makes [`LeanCluster::fetch`] return an error after dropping that
//! entry's sockets (a socket with an unread or late reply is never reused). The caller then
//! serves this fetch through the regular redis-rs path, which handles redirections and retries, and
//! the slot table is refreshed (rate-limited). Writes never use this path.
//!
//! Not supported (the lean path disables itself and everything goes through redis-rs): TLS
//! (`rediss://`) and authenticated endpoints.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Number of hash slots in a Redis-protocol cluster.
pub(crate) const NUM_SLOTS: usize = 16384;
/// `owner` value for a slot no node serves.
const UNCOVERED: u16 = u16::MAX;
/// Minimum interval between two slot-table refreshes (bounds refresh storms during an outage).
const REFRESH_MIN_INTERVAL: Duration = Duration::from_millis(100);
/// Read granularity: the buffer always has at least this much free space before a read.
const READ_CHUNK: usize = 64 * 1024;

// ------------------------------------------------------------------------------------------------
// Hash slots
// ------------------------------------------------------------------------------------------------

/// CRC16/XMODEM, as used by Redis Cluster key hashing.
pub(crate) fn crc16(data: &[u8]) -> u16 {
    let mut crc: u16 = 0;
    for &b in data {
        crc ^= (b as u16) << 8;
        for _ in 0..8 {
            crc = if crc & 0x8000 != 0 { (crc << 1) ^ 0x1021 } else { crc << 1 };
        }
    }
    crc
}

/// Redis Cluster hash slot of a key (honours a non-empty `{hash tag}`).
pub(crate) fn key_slot(key: &[u8]) -> u16 {
    let hashed = match key.iter().position(|&c| c == b'{') {
        Some(open) => match key[open + 1..].iter().position(|&c| c == b'}') {
            Some(len) if len > 0 => &key[open + 1..open + 1 + len],
            _ => key,
        },
        None => key,
    };
    crc16(hashed) % NUM_SLOTS as u16
}

/// Append the decimal digits of `n`.
fn push_decimal(buf: &mut Vec<u8>, mut n: usize) {
    let mut tmp = [0u8; 20];
    let mut i = tmp.len();
    loop {
        i -= 1;
        tmp[i] = b'0' + (n % 10) as u8;
        n /= 10;
        if n == 0 {
            break;
        }
    }
    buf.extend_from_slice(&tmp[i..]);
}

// ------------------------------------------------------------------------------------------------
// Slot table
// ------------------------------------------------------------------------------------------------

/// Slot → primary node, from `CLUSTER SLOTS`.
#[derive(Debug)]
pub(crate) struct SlotMap {
    /// Primary node addresses, `host:port`.
    pub(crate) nodes: Vec<String>,
    /// For each slot, an index into `nodes` (`UNCOVERED` if no node serves it).
    owner: Vec<u16>,
    /// Bumped on every refresh; pooled entries reconnect when it changes.
    pub(crate) generation: u64,
}

impl SlotMap {
    fn empty(generation: u64) -> Self {
        Self { nodes: Vec::new(), owner: vec![UNCOVERED; NUM_SLOTS], generation }
    }

    /// Build from a `CLUSTER SLOTS` reply: `[[start, end, [host, port, id, ...], replicas...], ...]`.
    /// An empty host means "the node you asked" (`fallback_host`).
    pub(crate) fn from_cluster_slots(
        v: &redis::Value,
        fallback_host: &str,
        generation: u64,
    ) -> Result<Self, String> {
        let int = |v: &redis::Value| -> Result<i64, String> {
            match v {
                redis::Value::Int(i) => Ok(*i),
                redis::Value::BulkString(b) => std::str::from_utf8(b)
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .ok_or_else(|| "CLUSTER SLOTS: bad integer".to_string()),
                other => Err(format!("CLUSTER SLOTS: expected integer, got {other:?}")),
            }
        };
        let text = |v: &redis::Value| -> Result<String, String> {
            match v {
                redis::Value::BulkString(b) => Ok(String::from_utf8_lossy(b).into_owned()),
                redis::Value::SimpleString(s) => Ok(s.clone()),
                redis::Value::Nil => Ok(String::new()),
                other => Err(format!("CLUSTER SLOTS: expected string, got {other:?}")),
            }
        };
        let redis::Value::Array(entries) = v else {
            return Err("CLUSTER SLOTS: expected an array".to_string());
        };
        let mut map = SlotMap::empty(generation);
        for e in entries {
            let redis::Value::Array(f) = e else {
                return Err("CLUSTER SLOTS: bad entry".to_string());
            };
            if f.len() < 3 {
                return Err("CLUSTER SLOTS: short entry".to_string());
            }
            let (start, end) = (int(&f[0])?, int(&f[1])?);
            let redis::Value::Array(primary) = &f[2] else {
                return Err("CLUSTER SLOTS: bad node".to_string());
            };
            if primary.len() < 2 {
                return Err("CLUSTER SLOTS: short node".to_string());
            }
            let mut host = text(&primary[0])?;
            if host.is_empty() || host == "?" {
                host = fallback_host.to_string();
            }
            let addr = format!("{host}:{}", int(&primary[1])?);
            if !(0..NUM_SLOTS as i64).contains(&start) || !(start..NUM_SLOTS as i64).contains(&end) {
                return Err(format!("CLUSTER SLOTS: bad range {start}-{end}"));
            }
            let node = match map.nodes.iter().position(|a| *a == addr) {
                Some(i) => i,
                None => {
                    map.nodes.push(addr);
                    map.nodes.len() - 1
                }
            };
            for s in start..=end {
                map.owner[s as usize] = node as u16;
            }
        }
        if map.nodes.is_empty() {
            return Err("CLUSTER SLOTS: no nodes".to_string());
        }
        Ok(map)
    }

    fn node_of(&self, slot: u16) -> Option<usize> {
        match self.owner[slot as usize] {
            UNCOVERED => None,
            n => Some(n as usize),
        }
    }
}

/// Ask each address in turn for `CLUSTER SLOTS`; the first good answer wins.
fn fetch_slot_map(addrs: &[String], timeout: Duration, generation: u64) -> Result<SlotMap, String> {
    let mut last_err = "no addresses".to_string();
    for addr in addrs {
        let attempt = (|| -> Result<SlotMap, String> {
            let client = redis::Client::open(format!("redis://{addr}")).map_err(|e| e.to_string())?;
            let mut conn = client.get_connection_with_timeout(timeout).map_err(|e| e.to_string())?;
            conn.set_read_timeout(Some(timeout)).map_err(|e| e.to_string())?;
            conn.set_write_timeout(Some(timeout)).map_err(|e| e.to_string())?;
            let v: redis::Value =
                redis::cmd("CLUSTER").arg("SLOTS").query(&mut conn).map_err(|e| e.to_string())?;
            let host = addr.rsplit_once(':').map_or(addr.as_str(), |(h, _)| h);
            SlotMap::from_cluster_slots(&v, host, generation)
        })();
        match attempt {
            Ok(m) => return Ok(m),
            Err(e) => last_err = format!("{addr}: {e}"),
        }
    }
    Err(last_err)
}

// ------------------------------------------------------------------------------------------------
// RESP reply parsing
// ------------------------------------------------------------------------------------------------

/// Why a lean fetch gave up (the caller falls back to redis-rs for this fetch).
#[derive(Debug)]
pub(crate) enum LeanError {
    /// The slot table is empty or does not cover a key's slot.
    NoRoute,
    /// The server answered with an error (`MOVED`, `ASK`, `CLUSTERDOWN`, ...).
    Server(String),
    /// Malformed or unexpected reply bytes.
    Protocol(String),
    /// Connect / read / write failure, including timeouts.
    Io(std::io::Error),
}

impl std::fmt::Display for LeanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LeanError::NoRoute => write!(f, "no route for a key's slot"),
            LeanError::Server(s) => write!(f, "server error: {s}"),
            LeanError::Protocol(s) => write!(f, "protocol error: {s}"),
            LeanError::Io(e) => write!(f, "i/o error: {e}"),
        }
    }
}

/// Find `"\r\n"` at or after `pos`: (line without CRLF, position after CRLF).
fn line(b: &[u8], pos: usize) -> Option<(&[u8], usize)> {
    let rest = b.get(pos..)?;
    let i = rest.windows(2).position(|w| w == b"\r\n")?;
    Some((&rest[..i], pos + i + 2))
}

fn parse_len(s: &[u8]) -> Result<i64, LeanError> {
    std::str::from_utf8(s)
        .ok()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| LeanError::Protocol(format!("bad length {:?}", String::from_utf8_lossy(s))))
}

/// Parse `n` consecutive bulk-string / nil replies (the replies to `n` pipelined `GET`s).
///
/// `Ok(None)` = not complete yet. `Ok(Some(end))` = complete; `spans` holds `(offset, len)` per
/// reply (`None` = nil) and `end` is the number of bytes consumed. An error reply anywhere aborts the
/// whole fetch (`LeanError::Server`) — the caller falls back rather than mixing partial results.
pub(crate) fn parse_bulk_seq(
    b: &[u8],
    n: usize,
    spans: &mut Vec<Option<(usize, usize)>>,
) -> Result<Option<usize>, LeanError> {
    spans.clear();
    let mut pos = 0;
    for _ in 0..n {
        let Some((head, p)) = line(b, pos) else { return Ok(None) };
        match head.first() {
            Some(b'$') => {}
            Some(b'-') => return Err(LeanError::Server(String::from_utf8_lossy(&head[1..]).into_owned())),
            _ => {
                return Err(LeanError::Protocol(format!(
                    "expected a bulk string, got {:?}",
                    String::from_utf8_lossy(&head[..head.len().min(40)])
                )))
            }
        }
        let len = parse_len(&head[1..])?;
        if len < 0 {
            spans.push(None);
            pos = p;
            continue;
        }
        let len = len as usize;
        if b.len() < p + len + 2 {
            return Ok(None);
        }
        if &b[p + len..p + len + 2] != b"\r\n" {
            return Err(LeanError::Protocol("bulk string not terminated by CRLF".to_string()));
        }
        spans.push(Some((p, len)));
        pos = p + len + 2;
    }
    Ok(Some(pos))
}

// ------------------------------------------------------------------------------------------------
// Pooled lean connections
// ------------------------------------------------------------------------------------------------

/// One pooled entry: a socket per node (opened lazily) plus reusable per-node buffers.
struct LeanConn {
    /// Slot-table generation the sockets belong to (`u64::MAX` = none / poisoned).
    generation: u64,
    socks: Vec<Option<TcpStream>>,
    /// Per node: the pipelined `GET`s of the current fetch.
    req: Vec<Vec<u8>>,
    /// Per node: positions (in the caller's slot list) of the keys sent to that node.
    pos: Vec<Vec<usize>>,
    /// Per node: read buffer and parsed value spans.
    rbuf: Vec<Vec<u8>>,
    spans: Vec<Vec<Option<(usize, usize)>>>,
    /// Scratch for building one key.
    key: Vec<u8>,
}

impl LeanConn {
    fn new() -> Self {
        Self {
            generation: u64::MAX,
            socks: Vec::new(),
            req: Vec::new(),
            pos: Vec::new(),
            rbuf: Vec::new(),
            spans: Vec::new(),
            key: Vec::with_capacity(64),
        }
    }

    /// Match the per-node vectors to `map`, dropping sockets opened for an older table.
    fn adopt(&mut self, map: &SlotMap) {
        if self.generation == map.generation {
            return;
        }
        let n = map.nodes.len();
        self.socks.clear();
        self.socks.resize_with(n, || None);
        self.req.resize_with(n, Vec::new);
        self.pos.resize_with(n, Vec::new);
        self.rbuf.resize_with(n, Vec::new);
        self.spans.resize_with(n, Vec::new);
        self.generation = map.generation;
    }

    /// Drop every socket: after any error a socket may hold unread or late replies.
    fn poison(&mut self) {
        for s in &mut self.socks {
            *s = None;
        }
        self.generation = u64::MAX;
    }
}

fn connect(addr: &str, timeout: Duration) -> std::io::Result<TcpStream> {
    let mut last = std::io::Error::new(std::io::ErrorKind::NotFound, format!("cannot resolve {addr}"));
    for sa in addr.to_socket_addrs()? {
        match TcpStream::connect_timeout(&sa, timeout) {
            Ok(s) => {
                s.set_nodelay(true)?;
                s.set_read_timeout(Some(timeout))?;
                s.set_write_timeout(Some(timeout))?;
                return Ok(s);
            }
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// Phase timings of one lean fetch (for the service's `fp_fetch` health line).
#[derive(Debug, Clone, Copy)]
pub(crate) struct LeanTiming {
    pub(crate) checked_out: Instant,
    pub(crate) prepared: Instant,
    pub(crate) received: Instant,
}

/// The lean fetch path of one `GarnetFpStore` (cluster mode).
pub(crate) struct LeanCluster {
    seeds: Vec<String>,
    map: RwLock<Arc<SlotMap>>,
    last_refresh: Mutex<Option<Instant>>,
    timeout: Duration,
    pool: Vec<Mutex<LeanConn>>,
    next: AtomicUsize,
    generation: AtomicU64,
}

impl LeanCluster {
    /// `seeds`: `host:port` of one or more cluster nodes. A failed initial table fetch is not an
    /// error: every fetch then falls back to redis-rs until a refresh succeeds.
    pub(crate) fn new(seeds: Vec<String>, timeout: Duration, pool_size: usize) -> Self {
        let me = Self {
            seeds,
            map: RwLock::new(Arc::new(SlotMap::empty(0))),
            last_refresh: Mutex::new(None),
            timeout,
            pool: (0..pool_size.max(1)).map(|_| Mutex::new(LeanConn::new())).collect(),
            next: AtomicUsize::new(0),
            generation: AtomicU64::new(0),
        };
        me.refresh();
        me
    }

    /// Current slot table.
    pub(crate) fn slot_map(&self) -> Arc<SlotMap> {
        match self.map.read() {
            Ok(g) => g.clone(),
            Err(p) => p.into_inner().clone(),
        }
    }

    /// Re-read `CLUSTER SLOTS` (from the known nodes, then the seeds). Rate-limited, and skipped if
    /// another thread is already refreshing (search threads never queue up behind a refresh).
    /// Returns whether a new table was installed.
    pub(crate) fn refresh(&self) -> bool {
        let mut last = match self.last_refresh.try_lock() {
            Ok(g) => g,
            Err(std::sync::TryLockError::Poisoned(p)) => p.into_inner(),
            Err(std::sync::TryLockError::WouldBlock) => return false,
        };
        if last.is_some_and(|t| t.elapsed() < REFRESH_MIN_INTERVAL) {
            return false;
        }
        *last = Some(Instant::now());
        let mut addrs = self.slot_map().nodes.clone();
        for s in &self.seeds {
            if !addrs.contains(s) {
                addrs.push(s.clone());
            }
        }
        let generation = self.generation.fetch_add(1, Ordering::Relaxed) + 1;
        match fetch_slot_map(&addrs, self.timeout, generation) {
            Ok(m) => {
                let m = Arc::new(m);
                match self.map.write() {
                    Ok(mut g) => *g = m,
                    Err(p) => *p.into_inner() = m,
                }
                true
            }
            Err(_) => false,
        }
    }

    /// Fetch `"{prefix}{slot}"` for every slot, in order. `out` gets one entry per slot (`None` for a
    /// missing key or a value `decode` rejects). On error nothing is guaranteed about `out`; the
    /// pooled entry's sockets are dropped and the caller should fall back (and may call
    /// [`Self::refresh`]).
    pub(crate) fn fetch<V>(
        &self,
        prefix: &str,
        slots: &[usize],
        out: &mut Vec<Option<V>>,
        decode: impl Fn(&[u8]) -> Option<V>,
        timing: &mut Option<LeanTiming>,
    ) -> Result<(), LeanError> {
        let map = self.slot_map();
        if map.nodes.is_empty() {
            return Err(LeanError::NoRoute);
        }
        let idx = self.next.fetch_add(1, Ordering::Relaxed) % self.pool.len();
        let mut guard = match self.pool[idx].lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        let checked_out = Instant::now();
        let conn = &mut *guard;
        conn.adopt(&map);
        let result = Self::fetch_on(conn, &map, self.timeout, prefix, slots, out, decode, checked_out, timing);
        if result.is_err() {
            conn.poison();
        }
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn fetch_on<V>(
        conn: &mut LeanConn,
        map: &SlotMap,
        timeout: Duration,
        prefix: &str,
        slots: &[usize],
        out: &mut Vec<Option<V>>,
        decode: impl Fn(&[u8]) -> Option<V>,
        checked_out: Instant,
        timing: &mut Option<LeanTiming>,
    ) -> Result<(), LeanError> {
        // 1. Route and encode: one pipelined GET per key into its node's buffer.
        for n in 0..map.nodes.len() {
            conn.req[n].clear();
            conn.pos[n].clear();
        }
        for (i, &slot) in slots.iter().enumerate() {
            conn.key.clear();
            conn.key.extend_from_slice(prefix.as_bytes());
            push_decimal(&mut conn.key, slot);
            let node = map.node_of(key_slot(&conn.key)).ok_or(LeanError::NoRoute)?;
            let req = &mut conn.req[node];
            req.extend_from_slice(b"*2\r\n$3\r\nGET\r\n$");
            push_decimal(req, conn.key.len());
            req.extend_from_slice(b"\r\n");
            req.extend_from_slice(&conn.key);
            req.extend_from_slice(b"\r\n");
            conn.pos[node].push(i);
        }
        let prepared = Instant::now();

        // 2. Send every node's batch before reading anything (one round trip of depth).
        for n in 0..map.nodes.len() {
            if conn.pos[n].is_empty() {
                continue;
            }
            if conn.socks[n].is_none() {
                conn.socks[n] = Some(connect(&map.nodes[n], timeout).map_err(LeanError::Io)?);
            }
            let sock = conn.socks[n].as_mut().expect("socket just ensured");
            sock.write_all(&conn.req[n]).map_err(LeanError::Io)?;
        }

        // 3. Read each node's replies: large reads, one parse pass once complete.
        for n in 0..map.nodes.len() {
            let count = conn.pos[n].len();
            if count == 0 {
                continue;
            }
            let sock = conn.socks[n].as_mut().expect("written above");
            let buf = &mut conn.rbuf[n];
            let spans = &mut conn.spans[n];
            let mut filled = 0usize;
            loop {
                if buf.len() < filled + READ_CHUNK {
                    buf.resize(filled + READ_CHUNK, 0);
                }
                let got = sock.read(&mut buf[filled..]).map_err(LeanError::Io)?;
                if got == 0 {
                    return Err(LeanError::Io(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "connection closed by server",
                    )));
                }
                filled += got;
                if let Some(end) = parse_bulk_seq(&buf[..filled], count, spans)? {
                    if end != filled {
                        return Err(LeanError::Protocol(format!(
                            "{} unexpected bytes after the replies",
                            filled - end
                        )));
                    }
                    break;
                }
            }
        }
        let received = Instant::now();

        // 4. Hand each value to the decoder, in the caller's order.
        out.clear();
        out.resize_with(slots.len(), || None);
        for n in 0..map.nodes.len() {
            for (j, span) in conn.spans[n].iter().enumerate() {
                if let Some((start, len)) = *span {
                    out[conn.pos[n][j]] = decode(&conn.rbuf[n][start..start + len]);
                }
            }
        }
        *timing = Some(LeanTiming { checked_out, prepared, received });
        Ok(())
    }
}

/// Seed `host:port` for the lean path from a configured endpoint, or `None` if the endpoint needs
/// something the lean path does not do (TLS, authentication, unix socket).
pub(crate) fn lean_seed(endpoint: &str) -> Option<String> {
    use redis::IntoConnectionInfo;
    let url = if endpoint.contains("://") { endpoint.to_string() } else { format!("redis://{endpoint}") };
    let info = url.as_str().into_connection_info().ok()?;
    if info.redis.password.is_some() || info.redis.username.is_some() {
        return None;
    }
    match info.addr {
        redis::ConnectionAddr::Tcp(host, port) => Some(format!("{host}:{port}")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_slots_match_redis() {
        assert_eq!(crc16(b"123456789"), 0x31C3); // CRC16/XMODEM check value
        assert_eq!(key_slot(b"foo"), 12182); // CLUSTER KEYSLOT foo
        assert_eq!(key_slot(b"somekey"), 11058);
        assert_eq!(key_slot(b"{user1000}.following"), key_slot(b"user1000"));
        assert_ne!(key_slot(b"foo{}{bar}"), key_slot(b"bar")); // empty tag: whole key hashed
    }

    #[test]
    fn decimal() {
        for n in [0usize, 7, 10, 999_999, 1_000_000, usize::MAX] {
            let mut b = Vec::new();
            push_decimal(&mut b, n);
            assert_eq!(b, n.to_string().into_bytes());
        }
    }

    #[test]
    fn parses_pipelined_replies_and_partials() {
        let reply = b"$3\r\nabc\r\n$-1\r\n$4\r\n\r\n\r\n\r\n";
        let mut spans = Vec::new();
        assert_eq!(parse_bulk_seq(reply, 3, &mut spans).unwrap(), Some(reply.len()));
        assert_eq!(spans, vec![Some((4, 3)), None, Some((18, 4))]);
        assert_eq!(&reply[18..22], b"\r\n\r\n"); // values containing CRLF are taken by length
        for cut in 0..reply.len() {
            assert!(parse_bulk_seq(&reply[..cut], 3, &mut spans).unwrap().is_none(), "cut {cut}");
        }
    }

    #[test]
    fn error_replies_and_garbage_are_errors() {
        let mut spans = Vec::new();
        match parse_bulk_seq(b"$1\r\na\r\n-MOVED 3999 10.0.0.2:6379\r\n", 2, &mut spans) {
            Err(LeanError::Server(s)) => assert!(s.starts_with("MOVED")),
            other => panic!("{other:?}"),
        }
        assert!(matches!(parse_bulk_seq(b"+OK\r\n", 1, &mut spans), Err(LeanError::Protocol(_))));
        assert!(matches!(parse_bulk_seq(b"$1\r\nab\r\n", 1, &mut spans), Err(LeanError::Protocol(_))));
    }

    #[test]
    fn slot_map_from_cluster_slots() {
        use redis::Value::{Array, BulkString, Int};
        let node = |h: &str, p: i64| Array(vec![BulkString(h.as_bytes().to_vec()), Int(p), BulkString(b"id".to_vec())]);
        let v = Array(vec![
            Array(vec![Int(0), Int(5460), node("10.0.0.1", 6379)]),
            Array(vec![Int(5461), Int(10922), node("10.0.0.2", 6379), node("10.0.0.9", 6379)]),
            Array(vec![Int(10923), Int(16383), node("", 7000)]),
        ]);
        let m = SlotMap::from_cluster_slots(&v, "seedhost", 3).unwrap();
        assert_eq!(m.nodes, vec!["10.0.0.1:6379", "10.0.0.2:6379", "seedhost:7000"]);
        assert_eq!(m.node_of(0), Some(0));
        assert_eq!(m.node_of(5461), Some(1)); // primary, not the replica
        assert_eq!(m.node_of(16383), Some(2));
        assert_eq!(m.generation, 3);
        let partial = Array(vec![Array(vec![Int(0), Int(100), node("a", 1)])]);
        assert_eq!(SlotMap::from_cluster_slots(&partial, "s", 0).unwrap().node_of(101), None);
    }

    #[test]
    fn seeds_from_endpoints() {
        assert_eq!(lean_seed("10.0.0.1:6379").as_deref(), Some("10.0.0.1:6379"));
        assert_eq!(lean_seed("redis://10.0.0.1:6380").as_deref(), Some("10.0.0.1:6380"));
        assert_eq!(lean_seed("redis://:secret@10.0.0.1:6379"), None);
    }

    /// End-to-end against a live cluster: `LEAN_TEST_CLUSTER=host:port cargo test -p diskann-bftree
    /// lean_fetch::tests::live -- --ignored --nocapture`. Writes keys `leantest:{0..300}`.
    #[test]
    #[ignore]
    fn live_matches_redis_rs_and_falls_back_on_moved() {
        let seed = std::env::var("LEAN_TEST_CLUSTER").expect("set LEAN_TEST_CLUSTER=host:port");
        let timeout = Duration::from_millis(1000);
        let client = redis::cluster::ClusterClient::new(vec![format!("redis://{seed}")]).unwrap();
        let mut rc = client.get_connection().unwrap();
        let slots: Vec<usize> = (0..300).collect();
        for &s in &slots {
            if s % 7 == 3 {
                let _: () = redis::cmd("DEL").arg(format!("leantest:{s}")).query(&mut rc).unwrap();
            } else {
                let val: Vec<u8> = (0..400u32).map(|i| ((i as usize * 31 + s) % 251) as u8).collect();
                let _: () = redis::cmd("SET").arg(format!("leantest:{s}")).arg(val).query(&mut rc).unwrap();
            }
        }
        let lean = LeanCluster::new(vec![seed.clone()], timeout, 2);
        assert!(lean.slot_map().nodes.len() >= 2, "expected a multi-node cluster");
        let mut out = Vec::new();
        let mut t = None;
        lean.fetch("leantest:", &slots, &mut out, |b| Some(b.to_vec()), &mut t).unwrap();
        let mut pipe = redis::cluster::cluster_pipe();
        for &s in &slots {
            pipe.cmd("GET").arg(format!("leantest:{s}"));
        }
        let expect: Vec<Option<Vec<u8>>> = pipe.query(&mut rc).unwrap();
        assert_eq!(out, expect, "lean fetch must match redis-rs byte for byte");
        assert!(t.is_some());

        // A stale table (every slot pointed at one node) must produce MOVED -> an error, never
        // wrong data; a refresh restores correct routing.
        {
            let real = lean.slot_map();
            let stale = SlotMap {
                nodes: real.nodes.clone(),
                owner: vec![0; NUM_SLOTS],
                generation: real.generation + 1000,
            };
            *lean.map.write().unwrap() = Arc::new(stale);
        }
        match lean.fetch("leantest:", &slots, &mut out, |b| Some(b.to_vec()), &mut t) {
            Err(LeanError::Server(s)) => assert!(s.starts_with("MOVED"), "{s}"),
            other => panic!("expected MOVED, got {other:?}"),
        }
        *lean.last_refresh.lock().unwrap() = None;
        assert!(lean.refresh());
        lean.fetch("leantest:", &slots, &mut out, |b| Some(b.to_vec()), &mut t).unwrap();
        assert_eq!(out, expect);
    }

    /// Through `GarnetFpStore` (the provider's real path), lean on vs off, then a live reshard:
    /// `LEAN_TEST_CLUSTER=host:port cargo test -p diskann-bftree lean_fetch::tests::store -- --ignored`.
    /// Needs `redis-cli` on PATH; moves 200 slots from the first node to the second.
    #[test]
    #[ignore]
    fn store_lean_matches_and_survives_reshard() {
        use crate::fp_store::{FpVectorStore, GarnetConfig, GarnetFpStore};
        let seed = std::env::var("LEAN_TEST_CLUSTER").expect("set LEAN_TEST_CLUSTER=host:port");
        let dim = 100;
        let cfg = |lean_fetch| GarnetConfig {
            endpoints: vec![seed.clone()],
            timeout_ms: 1000,
            pool_size: 4,
            breaker_threshold: 20,
            key_prefix: "leanstore:".to_string(),
            cluster: true,
            lean_fetch,
        };
        let lean = GarnetFpStore::<f32>::connect(&cfg(true), dim, 5000, 1).unwrap();
        let plain = GarnetFpStore::<f32>::connect(&cfg(false), dim, 5000, 1).unwrap();
        assert!(lean.lean.is_some() && plain.lean.is_none());
        let ids: Vec<usize> = (0..2000).collect();
        for &i in &ids {
            if i % 11 != 5 {
                let v: Vec<f32> = (0..dim).map(|d| (i * 1000 + d) as f32).collect();
                plain.set_vector_sync(i, &v).unwrap();
            } else {
                plain.delete_vector(i);
            }
        }
        let fetch = |s: &GarnetFpStore<f32>| {
            let mut all = Vec::new();
            for chunk in ids.chunks(100) {
                let mut out = Vec::new();
                s.get_many(chunk, &mut out).unwrap();
                all.extend(out);
            }
            all
        };
        let stats = crate::fp_store::fp_fetch_stats;
        let before = stats();
        let expect = fetch(&plain);
        assert_eq!(fetch(&lean), expect);
        assert_eq!(expect.iter().filter(|v| v.is_none()).count(), ids.iter().filter(|i| *i % 11 == 5).count());
        let mid = stats().since(&before);
        assert_eq!(mid.lean_fetches, 20, "every lean-store fetch served by the lean path");
        assert_eq!(mid.lean_fallbacks, 0);

        // Move 200 slots between nodes behind the store's back: the next lean fetches hit MOVED,
        // fall back to redis-rs (correct data), refresh the table, and go lean again.
        let map = lean.lean.as_ref().unwrap().slot_map();
        let id_of = |addr: &str| {
            let (h, p) = addr.rsplit_once(':').unwrap();
            let o = std::process::Command::new("redis-cli").args(["-h", h, "-p", p, "CLUSTER", "MYID"]).output().unwrap();
            String::from_utf8(o.stdout).unwrap().trim().to_string()
        };
        let (from, to) = (id_of(&map.nodes[0]), id_of(&map.nodes[1]));
        let st = std::process::Command::new("redis-cli")
            .args(["--cluster", "reshard", &map.nodes[0], "--cluster-from", &from, "--cluster-to", &to,
                   "--cluster-slots", "200", "--cluster-yes"])
            .output()
            .unwrap();
        assert!(st.status.success(), "reshard failed: {}", String::from_utf8_lossy(&st.stderr));
        std::thread::sleep(Duration::from_millis(300));
        let before = stats();
        assert_eq!(fetch(&lean), expect, "correct data across a reshard");
        assert_eq!(fetch(&lean), expect);
        let after = stats().since(&before);
        assert!(after.lean_fallbacks >= 1, "reshard should have forced at least one fallback");
        assert!(after.lean_fetches >= 1, "lean path should recover after the refresh");
        assert!(lean.is_available());
    }

    /// A node going down and coming back. `LEAN_TEST_CLUSTER=host:port LEAN_TEST_RESTART="<shell
    /// command that restarts the stopped node>"`; the test stops the last node in the slot table with
    /// `SHUTDOWN NOSAVE` (so the cluster must keep its data elsewhere or be a throwaway).
    #[test]
    #[ignore]
    fn store_node_down_then_back() {
        use crate::fp_store::{FpVectorStore, GarnetConfig, GarnetFpStore};
        let seed = std::env::var("LEAN_TEST_CLUSTER").expect("set LEAN_TEST_CLUSTER=host:port");
        let restart = std::env::var("LEAN_TEST_RESTART").expect("set LEAN_TEST_RESTART");
        let dim = 100;
        let cfg = GarnetConfig {
            endpoints: vec![seed.clone()],
            timeout_ms: 300,
            pool_size: 2,
            breaker_threshold: 1000,
            key_prefix: "leandown:".to_string(),
            cluster: true,
            lean_fetch: true,
        };
        let store = GarnetFpStore::<f32>::connect(&cfg, dim, 5000, 1).unwrap();
        let ids: Vec<usize> = (0..100).collect();
        for &i in &ids {
            store.set_vector_sync(i, &vec![i as f32; dim]).unwrap();
        }
        let mut out = Vec::new();
        store.get_many(&ids, &mut out).unwrap();
        assert!(out.iter().all(|v| v.is_some()));

        let victim = store.lean.as_ref().unwrap().slot_map().nodes.last().unwrap().clone();
        let (h, p) = victim.rsplit_once(':').unwrap();
        let _ = std::process::Command::new("redis-cli").args(["-h", h, "-p", p, "SHUTDOWN", "NOSAVE"]).output();
        std::thread::sleep(Duration::from_millis(200));
        let t = Instant::now();
        let r = store.get_many(&ids, &mut out);
        assert!(r.is_err(), "a fetch touching a dead node must fail (caller degrades), got {r:?}");
        assert!(t.elapsed() < Duration::from_secs(5), "failure must be bounded by timeouts: {:?}", t.elapsed());

        let st = std::process::Command::new("sh").args(["-c", &restart]).status().unwrap();
        assert!(st.success());
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            std::thread::sleep(Duration::from_millis(300));
            // The stopped node lost its keys (SHUTDOWN NOSAVE): rewrite, then read back.
            let rewrote = ids.iter().all(|&i| store.set_vector_sync(i, &vec![i as f32; dim]).is_ok());
            if rewrote && store.get_many(&ids, &mut out).is_ok() && out.iter().all(|v| v.is_some()) {
                break;
            }
            assert!(Instant::now() < deadline, "store did not recover after the node came back");
        }
        let before = crate::fp_store::fp_fetch_stats();
        store.get_many(&ids, &mut out).unwrap();
        let after = crate::fp_store::fp_fetch_stats().since(&before);
        assert_eq!((after.lean_fetches, after.lean_fallbacks), (1, 0), "lean path back in use");
    }
}

#[cfg(test)]
mod bench {
    /// Rough single-thread timing, lean vs redis-rs, through `GarnetFpStore`:
    /// `LEAN_TEST_CLUSTER=host:port cargo test --release -p diskann-bftree lean_fetch::bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn lean_vs_redis_rs() {
        use crate::fp_store::{FpVectorStore, GarnetConfig, GarnetFpStore};
        let seed = std::env::var("LEAN_TEST_CLUSTER").expect("set LEAN_TEST_CLUSTER=host:port");
        let dim = 100;
        let cfg = |lean_fetch| GarnetConfig {
            endpoints: vec![seed.clone()],
            timeout_ms: 1000,
            pool_size: 1,
            breaker_threshold: 20,
            key_prefix: "leanbench:".to_string(),
            cluster: true,
            lean_fetch,
        };
        let lean = GarnetFpStore::<f32>::connect(&cfg(true), dim, 100_000, 1).unwrap();
        let plain = GarnetFpStore::<f32>::connect(&cfg(false), dim, 100_000, 1).unwrap();
        let ids: Vec<usize> = (0..100).map(|i| i * 997 % 100_000).collect();
        for &i in &ids {
            plain.set_vector_sync(i, &vec![i as f32; dim]).unwrap();
        }
        for (name, s) in [("redis-rs", &plain), ("lean", &lean), ("redis-rs", &plain), ("lean", &lean)] {
            let mut out = Vec::new();
            for _ in 0..200 {
                s.get_many(&ids, &mut out).unwrap();
            }
            let n = 3000;
            let t = std::time::Instant::now();
            for _ in 0..n {
                s.get_many(&ids, &mut out).unwrap();
            }
            println!("{name:>8}: {:.1} us per 100-key fetch", t.elapsed().as_secs_f64() * 1e6 / n as f64);
        }
    }
}
