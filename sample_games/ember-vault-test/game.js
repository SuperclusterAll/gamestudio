'use strict';
const {Game,W,H}=window.EmberVault;
const $=id=>document.getElementById(id),canvas=$('game'),ctx=canvas.getContext('2d');
let game=new Game(),last=performance.now(),shown='',aim=null,firing=false,auto=false,sound=false,audio=null,touchOrigin=null,touchVector={x:0,y:0},dashTap=false;
let fireTap=false;
const keys=new Set();
const clock=t=>`${String(Math.floor(t/60)).padStart(2,'0')}:${String(Math.floor(t%60)).padStart(2,'0')}`;
const pos=e=>{const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*W/r.width,y:(e.clientY-r.top)*H/r.height};};
function tone(f=440,d=.06,type='sine',volume=.025){if(!sound||!audio)return;const osc=audio.createOscillator(),v=audio.createGain();osc.type=type;osc.frequency.value=f;osc.connect(v);v.connect(audio.destination);v.gain.setValueAtTime(volume,audio.currentTime);v.gain.exponentialRampToValueAtTime(.001,audio.currentTime+d);osc.start();osc.stop(audio.currentTime+d);}
function start(){game=new Game();game.start();shown='';keys.clear();firing=false;aim=null;canvas.focus();tone(330,.15);}
$('restart').onclick=()=>{if(game.mode==='playing'||game.mode==='paused'){game.mode='confirm';shown='';}else start();};
$('pause').onclick=()=>game.pause();
$('sound').onclick=()=>{sound=!sound;if(sound){audio=audio||new(window.AudioContext||window.webkitAudioContext)();audio.resume();tone();}$('sound').textContent=`소리 ${sound?'ON':'OFF'}`;};
function toggleAuto(){auto=!auto;aim=null;for(const id of ['autoFire','assist'])$(id).textContent=`자동 사격 ${auto?'ON':'OFF'}`;}
$('autoFire').onclick=toggleAuto;$('assist').onclick=toggleAuto;
$('touchDash').onpointerdown=e=>{e.preventDefault();dashTap=true;};
document.addEventListener('keydown',e=>{
  if(['Space','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight'].includes(e.code))e.preventDefault();
  keys.add(e.code);if(e.repeat)return;
  if(e.code==='Space')fireTap=true;
  if(e.code==='KeyP'||e.code==='Escape'){game.pause();shown='';}
  if(game.mode==='upgrade'&&['Digit1','Digit2','Digit3'].includes(e.code)){game.chooseUpgrade(Number(e.code.slice(-1))-1);tone(600,.14);}
});
document.addEventListener('keyup',e=>keys.delete(e.code));
window.addEventListener('blur',()=>{keys.clear();firing=false;touchVector={x:0,y:0};if(game.mode==='playing')game.pause();});
canvas.addEventListener('pointerdown',e=>{e.preventDefault();canvas.focus();canvas.setPointerCapture(e.pointerId);const p=pos(e);if(e.pointerType==='touch'){touchOrigin=p;aim=null;}else{aim=p;firing=true;}});
canvas.addEventListener('pointermove',e=>{const p=pos(e);if(e.pointerType==='touch'&&touchOrigin){touchVector={x:(p.x-touchOrigin.x)/50,y:(p.y-touchOrigin.y)/50};}else if(e.pointerType!=='touch')aim=p;});
function release(){firing=false;touchOrigin=null;touchVector={x:0,y:0};}
canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('contextmenu',e=>e.preventDefault());
function modal(){
  if(shown===game.mode)return;shown=game.mode;$('overlay').hidden=game.mode==='playing';hud();
  const b=$('modal');
  if(game.mode==='title'){b.innerHTML='<div class="kicker">A SMALL ROGUELIKE ADVENTURE</div><div class="ornament">◈</div><h2>잿불 금고</h2><p>잠든 금고의 여섯 방을 내려가 수호자를 쓰러뜨리세요.<br>전투마다 세 유물 중 하나를 선택하고,<br>위험한 보상과 회복의 경로 사이에서 결정하세요.</p><button class="primary" id="startGame">금고로 내려가기 →</button><div class="tiny">WASD 이동 · 클릭 / SPACE 사격 · SHIFT 회피</div>';$('startGame').onclick=start;}
  else if(game.mode==='paused'){b.innerHTML='<div class="kicker">TAKE A BREATH</div><h2>잠시 멈춤</h2><p>당신의 탐험은 기다립니다.</p><button class="primary" id="resume">계속 탐험</button>';$('resume').onclick=()=>{game.pause();canvas.focus();};}
  else if(game.mode==='confirm'){b.innerHTML='<h2>새 탐험을 시작할까요?</h2><p>현재 탐험의 유물과 진행도는 초기화됩니다.</p><button class="primary" id="confirm">새 탐험</button> <button class="small" id="cancel">돌아가기</button>';$('confirm').onclick=start;$('cancel').onclick=()=>{game.mode='playing';};}
  else if(game.mode==='upgrade'){
    b.innerHTML=`<div class="kicker">CHAMBER ${game.room} CLEARED</div><h2>하나의 힘을 선택하세요</h2><p>선택한 유물은 이번 탐험이 끝날 때까지 유지됩니다.</p><div class="cards">${game.choices.map((u,i)=>`<button class="card" data-upgrade="${i}"><small>0${i+1} / ${u.tag}</small><b>${u.name}</b><span>${u.desc}</span></button>`).join('')}</div>`;
    b.querySelectorAll('[data-upgrade]').forEach(el=>el.onclick=()=>{game.chooseUpgrade(Number(el.dataset.upgrade));tone(660,.2);});
  }else if(game.mode==='route'){
    b.innerHTML=`<div class="kicker">CHOOSE YOUR DESCENT</div><h2>${game.room===5?'수호자의 문 앞에서':'갈림길'}</h2><p>${game.room===5?'마지막 전투를 준비하세요. 어느 경로든 수호자를 만납니다.':'강해질 것인가, 살아남을 것인가.'}</p><div class="cards"><button class="card" id="danger"><small>RISK / 위험</small><b>잿불의 길</b><span>적 2명 추가 · 방 클리어 후 공격력 +12%<br>보스전에서는 추가 적·보상 없음</span></button><button class="card" id="rest"><small>RECOVERY / 회복</small><b>샘의 길</b><span>체력 22 회복 · 기본 전투</span></button></div>`;
    for(const kind of ['danger','rest'])$(kind).onclick=()=>{game.chooseRoute(kind);canvas.focus();};
  }else if(game.mode==='won'||game.mode==='lost'){
    const won=game.mode==='won';b.innerHTML=`<div class="kicker">${won?'VAULT OPENED':'THE EMBER FADES'}</div><div class="ornament">${won?'✧':'◇'}</div><h2>${won?'금고가 열렸습니다':'잿불이 꺼졌습니다'}</h2><p>${won?'수호자를 무너뜨리고 잿불의 핵을 되찾았습니다.':'유물은 사라져도, 다음 선택은 남습니다.'}<br>${game.room} / 6 방 · ${game.kills} 처치 · ${clock(game.time)}<br>점수 ${game.score.toLocaleString()}</p><button class="primary" id="again">새로운 탐험 →</button>`;$('again').onclick=start;
  }
}
function circle(x,y,r,fill,stroke){ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);if(fill){ctx.fillStyle=fill;ctx.fill();}if(stroke){ctx.strokeStyle=stroke;ctx.stroke();}}
function diamond(x,y,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x,y-r);ctx.lineTo(x+r,y);ctx.lineTo(x,y+r);ctx.lineTo(x-r,y);ctx.closePath();ctx.fill();}
function render(){
  ctx.clearRect(0,0,W,H);ctx.save();
  if(game.shake)ctx.translate(Math.sin(game.time*91)*game.shake,Math.cos(game.time*77)*game.shake);
  ctx.fillStyle='#141c21';ctx.fillRect(0,0,W,H);
  const g=ctx.createRadialGradient(480,290,30,480,290,560);g.addColorStop(0,'#26383c');g.addColorStop(1,'#10171d');ctx.fillStyle=g;ctx.fillRect(30,30,900,540);
  for(let y=36;y<565;y+=48)for(let x=36;x<925;x+=48){ctx.strokeStyle='#c2ad8410';ctx.strokeRect(x,y,48,48);if((x+y)%96===24){ctx.fillStyle='#0000000a';ctx.fillRect(x,y,48,48);}}
  ctx.strokeStyle='#c6a67655';ctx.lineWidth=2;ctx.strokeRect(30,30,900,540);ctx.strokeStyle='#c6a67615';ctx.strokeRect(20,20,920,560);
  for(const x of [105,855])for(const y of [100,500]){const glow=ctx.createRadialGradient(x,y,0,x,y,95);glow.addColorStop(0,'#e9a75522');glow.addColorStop(1,'#e9a75500');ctx.fillStyle=glow;ctx.fillRect(x-95,y-95,190,190);diamond(x,y,4,'#e9b878');}
  ctx.save();ctx.translate(480,300);ctx.rotate(Math.PI/4);ctx.strokeStyle='#d9bc7822';ctx.strokeRect(-75,-75,150,150);ctx.strokeRect(-66,-66,132,132);ctx.restore();
  for(const o of game.obstacles){ctx.fillStyle='#0005';ctx.fillRect(o.x+8,o.y+12,o.w,o.h);ctx.fillStyle='#29353b';ctx.fillRect(o.x,o.y,o.w,o.h);ctx.strokeStyle='#927e5555';ctx.strokeRect(o.x,o.y,o.w,o.h);ctx.fillStyle='#3f4a4b';ctx.fillRect(o.x+6,o.y+6,o.w-12,8);diamond(o.x+o.w/2,o.y+o.h/2,9,'#b79a6033');}
  for(const h of game.hazards){circle(h.x,h.y,h.r,h.fired?'#ed5c6444':'#ec6e7215','#ed7475aa');circle(h.x,h.y,h.r*(1-Math.max(0,h.t)),null,'#ed7475aa');}
  for(const b of game.bullets){ctx.strokeStyle=b.enemy?'#e97e79':'#e6c38b';ctx.lineWidth=b.enemy?3:2;ctx.beginPath();ctx.moveTo(b.x,b.y);ctx.lineTo(b.x-b.vx*.023,b.y-b.vy*.023);ctx.stroke();circle(b.x,b.y,b.r,b.enemy?'#ff9e8d':'#ffe0ab');}
  for(const e of game.enemies){
    circle(e.x+3,e.y+8,e.r,'#0005');
    if(e.type==='boss'){
      circle(e.x,e.y,47,null,'#bd7f6577');ctx.save();ctx.translate(e.x,e.y);ctx.rotate(game.time*.3);for(let i=0;i<8;i++){ctx.rotate(Math.PI/4);diamond(0,39,6,'#e2a76f');}ctx.restore();
      diamond(e.x,e.y,e.r,e.flash?'#fff3dd':'#bd765e');diamond(e.x,e.y,22,'#382934');diamond(e.x,e.y,9,'#fff0c4');
      ctx.fillStyle='#211b21';ctx.fillRect(260,48,440,7);ctx.fillStyle='#cc8d72';ctx.fillRect(260,48,440*Math.max(0,e.hp/e.maxHp),7);ctx.fillStyle='#e4c8a5';ctx.font='11px monospace';ctx.textAlign='center';ctx.fillText(e.hp/e.maxHp<.5?'금고의 수호자 / 분노':'금고의 수호자',480,76);
    }else{
      const color=e.flash?'#fff4cf':e.type==='archer'?'#ac98c8':e.type==='charger'?'#e0ac70':'#be7e76';
      if(e.type==='charger'){diamond(e.x,e.y,19,color);if(e.phase==='tell'){ctx.strokeStyle='#eda87588';ctx.lineWidth=18;ctx.beginPath();ctx.moveTo(e.x,e.y);ctx.lineTo(e.x+e.dx*180,e.y+e.dy*180);ctx.stroke();}}
      else{ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(e.x,e.y-18);ctx.lineTo(e.x+16,e.y+12);ctx.lineTo(e.x-16,e.y+12);ctx.closePath();ctx.fill();}
      circle(e.x,e.y,5,'#22242b');circle(e.x,e.y-2,2,'#ffe2b2');
      if(e.hp<e.maxHp){ctx.fillStyle='#0007';ctx.fillRect(e.x-17,e.y-26,34,3);ctx.fillStyle=color;ctx.fillRect(e.x-17,e.y-26,34*e.hp/e.maxHp,3);}
    }
  }
  const p=game.player;if(!(p.inv>.25&&Math.floor(game.time*18)%2)){
    if(p.dashTime>0)circle(p.x,p.y,23,'#93c9c622');circle(p.x+2,p.y+7,15,'#0005');
    ctx.save();ctx.translate(p.x,p.y);ctx.rotate(p.angle);ctx.fillStyle='#92bab7';ctx.beginPath();ctx.moveTo(-17,-12);ctx.lineTo(8,-9);ctx.lineTo(12,0);ctx.lineTo(8,9);ctx.lineTo(-17,12);ctx.lineTo(-10,0);ctx.fill();circle(0,0,10,'#e6d2af');ctx.fillStyle='#333e41';ctx.fillRect(8,-4,21,8);ctx.fillStyle='#ebc38c';ctx.fillRect(22,-3,7,6);ctx.restore();
    if(p.inv>0)circle(p.x,p.y,22,null,'#d5b77755');
  }
  for(const f of game.particles){ctx.globalAlpha=Math.min(1,f.t*3);ctx.fillStyle=f.color;ctx.fillRect(f.x,f.y,3,3);}ctx.globalAlpha=1;
  if(aim&&game.mode==='playing'){ctx.strokeStyle='#dbc49388';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(aim.x-7,aim.y);ctx.lineTo(aim.x+7,aim.y);ctx.moveTo(aim.x,aim.y-7);ctx.lineTo(aim.x,aim.y+7);ctx.stroke();}
  if(game.transition>0&&game.mode==='playing'){ctx.globalAlpha=Math.min(1,game.transition);ctx.fillStyle='#f1d8b0';ctx.textAlign='center';ctx.font='20px serif';ctx.fillText(game.room===6?'VI / 수호자의 방':`${String(game.room).padStart(2,'0')} / ${game.roomKind==='danger'?'잿불의 길':game.roomKind==='rest'?'샘의 길':'금고의 입구'}`,480,430);ctx.globalAlpha=1;}
  if(touchOrigin){circle(touchOrigin.x,touchOrigin.y,45,'#ffffff09','#ffffff33');circle(touchOrigin.x+Math.max(-1,Math.min(1,touchVector.x))*35,touchOrigin.y+Math.max(-1,Math.min(1,touchVector.y))*35,16,'#ffffff22');}
  ctx.restore();
}
let uiTick=0;
function hud(){const p=game.player;$('hp').textContent=`${Math.ceil(p.hp)} / ${p.maxHp}`;$('hpbar').style.width=`${p.hp/p.maxHp*100}%`;$('dash').textContent=p.dashCooldown>0?`${p.dashCooldown.toFixed(1)}s`:'준비';$('dashbar').style.width=`${(1-p.dashCooldown/p.dashInterval)*100}%`;
  $('roomLabel').textContent=game.room?`CHAMBER ${String(game.room).padStart(2,'0')} / 06`:'탐험 준비';$('score').textContent=String(game.score).padStart(4,'0');$('stats').textContent=`${game.kills} / ${clock(game.time)}`;
  $('map').innerHTML=Array.from({length:6},(_,i)=>`<div class="dot ${i+1<game.room?'past':i+1===game.room?'current':''}" title="${i+1}번 방"></div>`).join('');
  $('objective').textContent=game.mode==='won'?'금고의 핵 회수 완료':game.room===6?'수호자를 쓰러뜨리세요':'모든 적을 처치하세요';$('detail').textContent=`남은 적 ${game.enemies.length} · 공격력 ${p.damage.toFixed(0)} · 탄환 ${p.shots}`;
  $('build').innerHTML=game.build.length?game.build.map(n=>`<li>${n}</li>`).join(''):'<li>아직 획득한 유물이 없습니다.</li>';$('seed').textContent=`SEED / ${game.seed}`;
}
function frame(now){const dt=(now-last)/1000;last=now;
  const x=(keys.has('KeyD')||keys.has('ArrowRight')?1:0)-(keys.has('KeyA')||keys.has('ArrowLeft')?1:0)+touchVector.x;
  const y=(keys.has('KeyS')||keys.has('ArrowDown')?1:0)-(keys.has('KeyW')||keys.has('ArrowUp')?1:0)+touchVector.y;
  game.step(dt,{x,y,aim:auto?null:aim,fire:fireTap||firing||auto||keys.has('Space'),dash:dashTap||keys.has('ShiftLeft')||keys.has('ShiftRight')});dashTap=false;fireTap=false;
  for(const e of game.events){if(e==='shoot')tone(190,.035,'triangle',.015);else if(e==='hurt')tone(75,.14,'sawtooth');else if(e==='kill')tone(480,.05);else if(e==='clear'||e==='won')tone(720,.3);else if(e==='dash')tone(300,.05);}
  game.events=[];render();modal();if(++uiTick%6===0)hud();requestAnimationFrame(frame);
}
modal();hud();requestAnimationFrame(frame);
