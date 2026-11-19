# Turbo Kart Rally 🏁

A 3D kart racing game you play in the browser. Race solo against CPU karts or online with up to 8 players.

- **Three.js** 3D graphics (loaded from a CDN, no build step)
- **Node.js + WebSocket** relay server with room codes and invite links
- Drift mini-turbo boosts (blue → orange → purple), boost pads, start boost
- Item boxes: ⚡ Turbo · 🛢️ Oil slick · 🚀 Homing rocket
- CPU karts (simulated by the host), minimap, 3-lap race, results
- Touch controls on mobile

## How to run

```bash
cd kart-racer
npm install
npm start
```

Open `http://localhost:3000` in a browser.

- **Solo**: click "혼자 달리기"
- **Multiplayer**: click "방 만들기", then share the room code or invite link
  - Same Wi-Fi: friends connect to `http://<my-PC-IP>:3000`
  - Over the internet: deploy to a Node host such as Render, Railway or Fly.io (`PORT` env var supported),
    or expose it temporarily with `ngrok http 3000`

## Controls

| Key | Action |
| --- | --- |
| ↑ / W | Accelerate |
| ↓ / S | Brake / reverse |
| ← → / A D | Steer |
| Shift / Space | Drift (steer + hold) |
| E / X | Use item |
| M | Mute |
| Esc | Leave |

Hold accelerate when the countdown shows "1" for a start boost.
