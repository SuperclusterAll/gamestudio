"""A complete local fallback: the pipeline always emits a playable game."""

from __future__ import annotations


def build_neon_drift_html(title: str = "Neon Drift") -> str:
    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A self-contained arcade survival game.">
  <title>{escaped_title}</title>
  <style>
    :root {{ color-scheme: dark; font-family: Inter, ui-sans-serif, system-ui, sans-serif; }}
    * {{ box-sizing: border-box; }} body {{ margin: 0; min-height: 100vh; display: grid; place-items: center;
      background: radial-gradient(circle at 50% 0%, #162651, #060914 65%); color: #e8f4ff; }}
    main {{ width: min(94vw, 860px); padding: 18px; }} .hud {{ display:flex; gap:12px; justify-content:space-between;
      align-items:center; flex-wrap:wrap; margin-bottom:10px; }} h1 {{ font-size:clamp(1.45rem,4vw,2.5rem); margin:0;
      letter-spacing:.06em; color:#8ff7ff; text-shadow:0 0 18px #1cbbdf; }} .pill {{ border:1px solid #35577c;
      padding:7px 12px; border-radius:999px; font-weight:700; }} canvas {{ width:100%; display:block; aspect-ratio:16/9;
      border:1px solid #416c9a; border-radius:16px; background:#080d1e; touch-action:none; box-shadow:0 15px 48px #000a; }}
    .instructions {{ margin:12px 0 0; color:#b6c9df; line-height:1.5; }} kbd, button {{ font:inherit; }} kbd {{ padding:2px 6px;
      border:1px solid #6b8aa7; border-radius:4px; }} button {{ cursor:pointer; color:#04101a; background:#8ff7ff;
      border:0; border-radius:8px; padding:8px 12px; font-weight:800; }} button:hover {{ background:#fff; }}
  </style>
</head>
<body><main>
  <div class="hud"><h1>{escaped_title}</h1><div class="pill" id="score" aria-live="polite">Score 0 · 60s</div>
  <button id="restart" type="button" aria-label="Restart game">Restart</button></div>
  <canvas id="game" width="960" height="540" aria-label="{escaped_title} game field. Move to collect sparks and avoid hunters."></canvas>
  <p class="instructions">Move with <kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> or arrow keys. On touch devices, drag on the game field. Collect sparks, avoid red hunters, survive 60 seconds.</p>
</main><script>
(() => {{
  const c = document.querySelector('#game'), x = c.getContext('2d'), scoreEl = document.querySelector('#score');
  const keys = new Set(); let pointer = null, state, last = 0;
  const clamp=(n,a,b)=>Math.max(a,Math.min(b,n)); const rnd=(a,b)=>a+Math.random()*(b-a);
  function reset() {{ state={{p:{{x:480,y:270,r:13}}, sparks:[], hunters:[], score:0,time:60,over:false,spawn:0}}; last=performance.now(); }}
  function addSpark() {{ state.sparks.push({{x:rnd(30,930),y:rnd(30,510),r:8,p:Math.random()*6.28}}); }}
  function addHunter() {{ let edge=Math.floor(Math.random()*4), p={{x:0,y:0}}; if(edge===0){{p.x=rnd(0,960)}} else if(edge===1){{p.x=960;p.y=rnd(0,540)}} else if(edge===2){{p.x=rnd(0,960);p.y=540}} else {{p.y=rnd(0,540)}}; state.hunters.push({{...p,r:rnd(10,18),v:rnd(52,86)}}); }}
  function update(dt) {{ if(state.over) return; const p=state.p, speed=240; let dx=(keys.has('ArrowRight')||keys.has('KeyD')?1:0)-(keys.has('ArrowLeft')||keys.has('KeyA')?1:0); let dy=(keys.has('ArrowDown')||keys.has('KeyS')?1:0)-(keys.has('ArrowUp')||keys.has('KeyW')?1:0);
    if(pointer) {{ dx=pointer.x-p.x;dy=pointer.y-p.y; const d=Math.hypot(dx,dy)||1; dx/=d;dy/=d; }} const d=Math.hypot(dx,dy)||1; p.x=clamp(p.x+dx/d*speed*dt,p.r,960-p.r); p.y=clamp(p.y+dy/d*speed*dt,p.r,540-p.r);
    state.spawn-=dt; if(state.spawn<=0){{ addSpark(); if(state.hunters.length<2+Math.floor((60-state.time)/8))addHunter(); state.spawn=.55; }}
    state.hunters.forEach(h=>{{ const vx=p.x-h.x,vy=p.y-h.y,dd=Math.hypot(vx,vy)||1; h.x+=vx/dd*h.v*dt;h.y+=vy/dd*h.v*dt; if(Math.hypot(vx,vy)<p.r+h.r)state.over=true; }});
    state.sparks=state.sparks.filter(s=>{{ if(Math.hypot(s.x-p.x,s.y-p.y)<p.r+s.r+3){{state.score+=10;return false}} return true }}); state.time=Math.max(0,state.time-dt); if(state.time===0)state.over=true;
  }}
  function circle(o,color,glow=0) {{ x.save(); x.shadowBlur=glow;x.shadowColor=color;x.fillStyle=color;x.beginPath();x.arc(o.x,o.y,o.r,0,7);x.fill();x.restore(); }}
  function draw() {{ x.clearRect(0,0,960,540); x.fillStyle='#080d1e';x.fillRect(0,0,960,540); x.strokeStyle='rgba(90,160,255,.12)'; for(let i=0;i<960;i+=48){{x.beginPath();x.moveTo(i,0);x.lineTo(i,540);x.stroke()}} for(let i=0;i<540;i+=48){{x.beginPath();x.moveTo(0,i);x.lineTo(960,i);x.stroke()}}
    state.sparks.forEach(s=>{{x.save();x.translate(s.x,s.y);x.rotate(s.p+=.06);x.fillStyle='#ffe46b';x.shadowBlur=18;x.shadowColor='#ffe46b';x.fillRect(-6,-6,12,12);x.restore()}}); state.hunters.forEach(h=>circle(h,'#ff5771',14)); circle(state.p,'#77f7ff',22);
    scoreEl.textContent=`Score ${{state.score}} · ${{Math.ceil(state.time)}}s`; if(state.over){{x.fillStyle='rgba(4,7,18,.75)';x.fillRect(0,0,960,540);x.textAlign='center';x.fillStyle='#fff';x.font='bold 44px system-ui';x.fillText(state.time===0?'TIME CLEARED':'RUN ENDED',480,238);x.font='24px system-ui';x.fillStyle='#8ff7ff';x.fillText(`Final score: ${{state.score}} — press R or Restart`,480,284)}}
  }}
  function frame(now) {{ const dt=Math.min(.04,(now-last)/1000);last=now;update(dt);draw();requestAnimationFrame(frame); }}
  // event.code, not event.key: e.key is the layout-dependent character, so W arrives as 'ㅈ' with a
  // Korean IME and as 'W' with Caps Lock, which silently killed WASD while the arrows kept working.
  addEventListener('keydown',e=>{{keys.add(e.code);if(e.code==='KeyR')reset()}});addEventListener('keyup',e=>keys.delete(e.code));
  function pos(e){{const r=c.getBoundingClientRect();return {{x:(e.clientX-r.left)*960/r.width,y:(e.clientY-r.top)*540/r.height}}}} c.addEventListener('pointerdown',e=>{{pointer=pos(e);c.setPointerCapture(e.pointerId)}});c.addEventListener('pointermove',e=>{{if(pointer)pointer=pos(e)}});c.addEventListener('pointerup',()=>pointer=null);document.querySelector('#restart').addEventListener('click',reset);
  reset();requestAnimationFrame(frame);
}})();</script></body></html>'''
