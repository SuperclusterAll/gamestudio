/* Ember Vault: deterministic game simulation, shared by browser and tests. */
(function (scope) {
  'use strict';
  const W=960,H=600;
  const clamp=(v,a,b)=>Math.max(a,Math.min(b,v));
  const dist=(a,b)=>Math.hypot(a.x-b.x,a.y-b.y);
  const upgrades=[
    {id:'power',name:'흑요석 탄두',tag:'화력',desc:'탄환 피해 +30%',apply:p=>p.damage*=1.3},
    {id:'rapid',name:'순환 장치',tag:'속도',desc:'사격 간격 −22%',apply:p=>p.interval*=.78},
    {id:'spread',name:'갈라진 불꽃',tag:'다중 사격',desc:'부채꼴 탄환 +2 · 탄환당 피해 −18%',apply:p=>{p.shots+=2;p.damage*=.82;}},
    {id:'vital',name:'잿빛 심장',tag:'생존',desc:'최대 체력 +25 · 체력 30 회복',apply:p=>{p.maxHp+=25;p.hp=Math.min(p.maxHp,p.hp+30);}},
    {id:'dash',name:'순간의 잔영',tag:'회피',desc:'대시 재사용 −25% · 이동 속도 +10%',apply:p=>{p.dashInterval*=.75;p.speed*=1.1;}},
    {id:'leech',name:'생명의 재',tag:'흡수',desc:'적 처치 시 체력 3 회복',apply:p=>p.leech+=3},
  ];
  class Game {
    constructor(seed=Date.now()) { this.seed=seed>>>0;this.rng=this.seed||1;this.reset(); }
    random(){let n=this.rng;n^=n<<13;n^=n>>>17;n^=n<<5;this.rng=n>>>0;return this.rng/4294967296;}
    reset(){
      this.rng=this.seed||1;this.mode='title';this.room=0;this.time=0;this.score=0;this.kills=0;this.shake=0;
      this.player={x:W/2,y:H-90,r:13,hp:100,maxHp:100,speed:220,damage:18,interval:.24,shots:1,
        leech:0,dashInterval:1.5,dashCooldown:0,dashTime:0,inv:0,fire:0,angle:-Math.PI/2};
      this.enemies=[];this.bullets=[];this.particles=[];this.obstacles=[];this.hazards=[];this.build=[];
      this.choices=[];this.events=[];this.path=[];this.roomKind='normal';this.transition=0;this.damageTaken=0;
    }
    start(){this.reset();this.room=1;this.loadRoom('normal');}
    pause(){if(this.mode==='playing')this.mode='paused';else if(this.mode==='paused')this.mode='playing';}
    enemy(type,x,y){const boss=type==='boss';return {type,x,y,r:boss?36:14,hp:boss?1000:32+this.room*9,
      maxHp:boss?1000:32+this.room*9,timer:this.random()*1.2+1,flash:0,phase:'move',dx:0,dy:0};}
    loadRoom(kind){
      this.roomKind=kind;this.path.push(kind);this.mode='playing';this.transition=1.3;
      Object.assign(this.player,{x:480,y:510,inv:1,dashTime:0,fire:0});
      this.bullets=[];this.hazards=[];this.enemies=[];this.obstacles=[];this.particles=[];
      if(kind==='rest')this.player.hp=Math.min(this.player.maxHp,this.player.hp+22);
      if(this.room===6){this.enemies.push(this.enemy('boss',480,165));return;}
      // Pillars never block the entry or spawn positions.
      const flip=this.random()>.5;
      this.obstacles=[{x:flip?255:300,y:235,w:60,h:85},{x:flip?645:600,y:235,w:60,h:85}];
      const count=3+this.room+(kind==='danger'?2:0);
      for(let i=0;i<count;i++){
        const x=95+(i%5)*190,y=95+Math.floor(i/5)*72;
        const type=this.room===1?'stalker':['stalker','archer','charger'][Math.floor(this.random()*3)];
        this.enemies.push(this.enemy(type,x,y));
      }
    }
    blocked(x,y,r){return x<36+r||x>W-36-r||y<36+r||y>H-36-r||this.obstacles.some(o=>x+r>o.x&&x-r<o.x+o.w&&y+r>o.y&&y-r<o.y+o.h);}
    move(o,dx,dy){if(!this.blocked(o.x+dx,o.y,o.r))o.x+=dx;if(!this.blocked(o.x,o.y+dy,o.r))o.y+=dy;}
    burst(x,y,color,count=10){for(let i=0;i<count;i++){let a=this.random()*Math.PI*2,s=35+this.random()*150;this.particles.push({x,y,vx:Math.cos(a)*s,vy:Math.sin(a)*s,t:.3+this.random()*.35,color});}}
    damagePlayer(amount){const p=this.player;if(p.inv>0||this.mode!=='playing')return;
      p.hp=Math.max(0,p.hp-amount);p.inv=.75;this.damageTaken+=amount;this.shake=8;this.burst(p.x,p.y,'#ff6b68');this.events.push('hurt');
      if(p.hp<=0){this.mode='lost';this.events.push('lost');}}
    shoot(x,y,a,enemy=false,damage=0,speed=500,r=4){this.bullets.push({x,y,vx:Math.cos(a)*speed,vy:Math.sin(a)*speed,enemy,damage,r,life:3});}
    hitEnemy(e,damage){if(e.hp<=0)return;e.hp-=damage;e.flash=.09;this.burst(e.x,e.y,'#f9c97b',3);
      if(e.hp<=0){this.score+=e.type==='boss'?1500:100;this.kills++;this.player.hp=Math.min(this.player.maxHp,this.player.hp+this.player.leech);this.burst(e.x,e.y,'#f9c97b',20);this.events.push('kill');}}
    completeRoom(){
      this.bullets=[];this.hazards=[];this.score+=250;
      if(this.room===6){this.mode='won';this.events.push('won');return;}
      this.mode='upgrade';this.choices=[...upgrades]; // Fisher-Yates; seeded and reproducible.
      for(let i=this.choices.length-1;i>0;i--){let j=Math.floor(this.random()*(i+1));[this.choices[i],this.choices[j]]=[this.choices[j],this.choices[i]];}
      this.choices=this.choices.slice(0,3);this.events.push('clear');
    }
    chooseUpgrade(index){if(this.mode!=='upgrade'||!this.choices[index])return false;
      const u=this.choices[index];u.apply(this.player);this.build.push(u.name);
      if(this.roomKind==='danger'){this.player.damage*=1.12;this.build.push('위험 보상 +12%');}
      this.mode='route';return true;}
    chooseRoute(kind){if(this.mode!=='route'||!['danger','rest'].includes(kind))return false;this.room++;this.loadRoom(kind);return true;}
    step(dt,input={}){
      dt=clamp(dt,0,.04);
      if(this.mode!=='playing')return;
      this.time+=dt;this.transition=Math.max(0,this.transition-dt);this.shake=Math.max(0,this.shake-dt*30);
      const p=this.player;p.inv=Math.max(0,p.inv-dt);p.fire-=dt;p.dashCooldown=Math.max(0,p.dashCooldown-dt);p.dashTime=Math.max(0,p.dashTime-dt);
      let dx=input.x||0,dy=input.y||0,len=Math.hypot(dx,dy);if(len>1){dx/=len;dy/=len;}
      if(input.dash&&p.dashCooldown===0&&len>0){p.dashTime=.19;p.inv=Math.max(p.inv,.24);p.dashCooldown=p.dashInterval;this.events.push('dash');}
      this.move(p,dx*p.speed*dt*(p.dashTime>0?3.8:1),dy*p.speed*dt*(p.dashTime>0?3.8:1));
      let target=input.aim;
      if(!target&&this.enemies.length)target=this.enemies.reduce((a,b)=>dist(a,p)<dist(b,p)?a:b);
      if(target)p.angle=Math.atan2(target.y-p.y,target.x-p.x);
      if(input.fire&&p.fire<=0){p.fire=p.interval;for(let i=0;i<p.shots;i++)this.shoot(p.x+Math.cos(p.angle)*20,p.y+Math.sin(p.angle)*20,p.angle+(i-(p.shots-1)/2)*.14,false,p.damage);this.events.push('shoot');}
      for(const e of this.enemies){
        if(e.hp<=0)continue;e.timer-=dt;e.flash=Math.max(0,e.flash-dt);
        let a=Math.atan2(p.y-e.y,p.x-e.x),d=dist(p,e);
        if(e.type==='boss'){
          const phase2=e.hp<e.maxHp*.5;
          this.move(e,Math.cos(a)*35*dt,Math.sin(a)*35*dt);
          if(e.timer<=0){e.timer=phase2?1.25:1.9;const n=phase2?16:10;
            for(let i=0;i<n;i++)this.shoot(e.x,e.y,a+i*Math.PI*2/n,true,14,phase2?190:150,6);
            this.hazards.push({x:p.x,y:p.y,r:65,t:1,active:.18,fired:false});}
        }else if(e.type==='archer'){
          if(d>290)this.move(e,Math.cos(a)*75*dt,Math.sin(a)*75*dt);
          else if(d<170)this.move(e,-Math.cos(a)*75*dt,-Math.sin(a)*75*dt);
          if(e.timer<=0){e.timer=1.8;this.shoot(e.x,e.y,a,true,12,210,5);}
        }else if(e.type==='charger'){
          if(e.phase==='move'){this.move(e,Math.cos(a)*75*dt,Math.sin(a)*75*dt);if(e.timer<=0){e.phase='tell';e.timer=.7;e.dx=Math.cos(a);e.dy=Math.sin(a);}}
          else if(e.phase==='tell'&&e.timer<=0){e.phase='charge';e.timer=.45;}
          else if(e.phase==='charge'){this.move(e,e.dx*440*dt,e.dy*440*dt);if(e.timer<=0){e.phase='move';e.timer=1.8;}}
        }else this.move(e,Math.cos(a)*(70+this.room*7)*dt,Math.sin(a)*(70+this.room*7)*dt);
        if(dist(p,e)<p.r+e.r)this.damagePlayer(e.type==='boss'?20:12);
      }
      for(const b of this.bullets){
        b.x+=b.vx*dt;b.y+=b.vy*dt;b.life-=dt;if(this.blocked(b.x,b.y,b.r))b.life=0;
        if(b.life<=0)continue;
        if(b.enemy){if(dist(b,p)<p.r+b.r){this.damagePlayer(b.damage);b.life=0;}}
        else for(const e of this.enemies){if(e.hp>0&&dist(e,b)<e.r+b.r){this.hitEnemy(e,b.damage);b.life=0;break;}}
      }
      for(const h of this.hazards){h.t-=dt;if(h.t<=0&&!h.fired){h.fired=true;this.burst(h.x,h.y,'#ff6b68',22);}
        if(h.fired){h.active-=dt;if(dist(h,p)<h.r+p.r)this.damagePlayer(18);}}
      this.hazards=this.hazards.filter(h=>h.t>0||h.active>0);this.bullets=this.bullets.filter(b=>b.life>0);
      this.enemies=this.enemies.filter(e=>e.hp>0);
      for(const f of this.particles){f.x+=f.vx*dt;f.y+=f.vy*dt;f.t-=dt;}this.particles=this.particles.filter(f=>f.t>0);
      if(this.mode==='playing'&&this.enemies.length===0)this.completeRoom();
      if(this.events.length>40)this.events=this.events.slice(-40);
    }
  }
  scope.EmberVault={Game,W,H,upgrades};
  if(typeof module!=='undefined')module.exports=scope.EmberVault;
})(typeof window!=='undefined'?window:globalThis);
