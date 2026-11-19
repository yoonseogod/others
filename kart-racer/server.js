// Turbo Kart Rally - multiplayer relay server
// Serves the static game and relays player state between players in the same room.
const http = require('http');
const fs = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');

const PORT = process.env.PORT || 3000;
const PUBLIC = path.join(__dirname, 'public');
const MIME = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript', '.css': 'text/css', '.png': 'image/png', '.svg': 'image/svg+xml' };

const server = http.createServer((req, res) => {
  let p = decodeURIComponent(req.url.split('?')[0]);
  if (p === '/') p = '/index.html';
  const file = path.normalize(path.join(PUBLIC, p));
  if (!file.startsWith(PUBLIC)) { res.writeHead(403); return res.end(); }
  fs.readFile(file, (err, data) => {
    if (err) { res.writeHead(404); return res.end('Not found'); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    res.end(data);
  });
});

const wss = new WebSocketServer({ server });
const rooms = new Map(); // code -> { players: Map<id, player>, host, state, botCount }
let nextId = 1;

function send(ws, msg) { if (ws.readyState === 1) ws.send(JSON.stringify(msg)); }
function broadcast(room, msg, exceptId) {
  const data = JSON.stringify(msg);
  for (const p of room.players.values()) if (p.id !== exceptId && p.ws.readyState === 1) p.ws.send(data);
}
function lobbyInfo(room) {
  return {
    t: 'lobby', host: room.host, state: room.state, bots: room.botCount,
    players: [...room.players.values()].map(p => ({ id: p.id, name: p.name, color: p.color, inRace: p.inRace })),
  };
}

wss.on('connection', ws => {
  const me = { id: 'p' + nextId++, ws, name: 'Player', color: '#e33', room: null, inRace: false };

  ws.on('message', raw => {
    let m;
    try { m = JSON.parse(raw); } catch { return; }
    const room = me.room && rooms.get(me.room);

    switch (m.t) {
      case 'ping':
        send(ws, { t: 'pong', c: m.c, s: Date.now() });
        break;

      case 'join': {
        if (me.room) return;
        const code = String(m.room || '').toUpperCase().replace(/[^A-Z0-9]/g, '').slice(0, 6);
        if (!code) return send(ws, { t: 'error', msg: '방 코드가 올바르지 않습니다.' });
        let r = rooms.get(code);
        if (!r) { r = { players: new Map(), host: me.id, state: 'lobby', botCount: 3 }; rooms.set(code, r); }
        if (r.players.size >= 8) return send(ws, { t: 'error', msg: '방이 가득 찼습니다 (최대 8명).' });
        me.name = String(m.name || 'Player').slice(0, 12);
        me.color = /^#[0-9a-f]{6}$/i.test(m.color) ? m.color : '#e33';
        me.room = code;
        r.players.set(me.id, me);
        send(ws, { t: 'welcome', id: me.id, room: code });
        broadcast(r, lobbyInfo(r));
        break;
      }

      case 'bots':
        if (room && room.host === me.id && room.state === 'lobby') {
          room.botCount = Math.max(0, Math.min(7, m.n | 0));
          broadcast(room, lobbyInfo(room));
        }
        break;

      case 'start': {
        if (!room || room.host !== me.id || room.state !== 'lobby') return;
        room.state = 'racing';
        const humans = [...room.players.values()];
        humans.forEach(p => (p.inRace = true));
        const bots = Math.min(room.botCount, 8 - humans.length);
        const grid = humans.map(p => ({ id: p.id, name: p.name, color: p.color }));
        const botColors = ['#3a7bd5', '#2ecc71', '#f1c40f', '#9b59b6', '#e67e22', '#1abc9c', '#ecf0f1'];
        const botNames = ['Bolt', 'Nitro', 'Comet', 'Drifty', 'Turbo', 'Zippy', 'Rocket'];
        for (let i = 0; i < bots; i++) grid.push({ id: 'b' + i, name: botNames[i] + ' (CPU)', color: botColors[i], bot: true });
        // shuffle grid order
        for (let i = grid.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [grid[i], grid[j]] = [grid[j], grid[i]]; }
        broadcast(room, { t: 'start', at: Date.now() + 4000, grid, host: room.host, seed: Math.floor(Math.random() * 1e9) });
        broadcast(room, lobbyInfo(room));
        break;
      }

      case 'state': {
        if (!room || room.state !== 'racing') return;
        // The host simulates CPU karts and may send state on their behalf.
        const id = m.id && room.host === me.id && /^b\d$/.test(m.id) ? m.id : me.id;
        broadcast(room, { t: 'state', id, s: m.s }, me.id);
        break;
      }

      case 'event':
        if (room && room.state === 'racing') broadcast(room, { t: 'event', id: me.id, e: m.e }, me.id);
        break;

      case 'lobby':
        if (room && room.host === me.id && room.state === 'racing') {
          room.state = 'lobby';
          for (const p of room.players.values()) p.inRace = false;
          broadcast(room, lobbyInfo(room));
        }
        break;
    }
  });

  ws.on('close', () => {
    const room = me.room && rooms.get(me.room);
    if (!room) return;
    room.players.delete(me.id);
    if (room.players.size === 0) return rooms.delete(me.room);
    if (room.host === me.id) room.host = room.players.keys().next().value;
    broadcast(room, { t: 'leave', id: me.id });
    broadcast(room, lobbyInfo(room));
  });
});

server.listen(PORT, () => console.log(`Turbo Kart Rally running at http://localhost:${PORT}`));
