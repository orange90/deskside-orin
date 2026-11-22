// Dependency-free browser-logic regression for offline release checks.
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const controls=new Map(),requests=[],storage=new Map();
class Node {constructor(){this.value='';this.checked=false;this.textContent='';this.hidden=false;this.style={};this.children=[];this.classList={toggle(){}};this.dataset={};}append(...nodes){this.children.push(...nodes)}replaceChildren(...nodes){this.children=nodes}setAttribute(k,v){this[k]=v}removeAttribute(k){delete this[k]}addEventListener(){}querySelector(){return null}getContext(){return {clearRect(){}}}showModal(){}close(){}}
const el=id=>{if(!controls.has(id))controls.set(id,new Node());return controls.get(id)};
const start=Date.parse('2026-11-11T00:00:00+08:00')/1000;
const fixture={date:'2026-11-11',today:'2026-11-11',first_date:'2026-11-10',start,end:start+86400,cutoff:start+3600,as_of:start+3600,totals:{sit:120},settings:{sit_minutes:50,stand_minutes:60},intervals:[],notes:[{kind:'reminder',body:'本地文字提醒',start,end:start,count:1}],active_reminders:[],tasks:[],camera:{ready:true,person:'sit',presence:'present',fps:6},hourly:Array.from({length:24},(_,hour)=>({hour,sit:0,stand:0,away:0,unknown:0,privacy:0}))};
const replies={'/api/dashboard':fixture,'/api/hardware':{latest:null,history:[]},'/api/models':{components:[{id:'asr',model:'Whisper small'}]},'/api/voice/start':{},'/api/voice/stop':{query:'今天坐了多久',answer:'今天坐了 2 分钟'}};
const inboxTabs = ['open', 'done', 'reminders'].map(view => {
  const tab = new Node();
  tab.dataset.inbox = view;
  return tab;
});
const todo = {id:'legacy / 1', body:'核对周报', completed:false};
replies['/api/todos'] = {items:[todo], next_cursor:null};
replies['/api/focus'] = {active:null, recent:[], server_now:start};
replies['/api/todos/completion'] = {...todo, completed:true};
const context=vm.createContext({AbortSignal,performance,document:{getElementById:el,createElement:()=>new Node(),createElementNS:()=>new Node(),createTextNode:text=>({textContent:text}),querySelectorAll:selector=>selector==='[data-inbox]'?inboxTabs:[],querySelector:el,addEventListener(){},hidden:false},localStorage:{getItem:key=>storage.get(key),setItem:(k,v)=>storage.set(k,v)},setInterval(){},setTimeout(){},clearTimeout(){},crypto:{randomUUID:()=> 'fixture'},fetch:async(path,options)=>{requests.push({path,body:options.body?JSON.parse(options.body):null});return{ok:true,json:async()=>replies[path.split('?')[0]]}},console});
const html=fs.readFileSync(0,'utf8');vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1],context);
(async()=>{await new Promise(setImmediate);await el('voice').onclick();await el('voice').onclick();await new Promise(setImmediate);
assert.deepEqual(requests.filter(x=>x.path.startsWith('/api/voice/')), [{path:'/api/voice/start',body:{}},{path:'/api/voice/stop',body:{id:'fixture',save:false}}]);assert.match(el('feedback').textContent,/识别到：今天坐了多久\n本机回复：今天坐了 2 分钟/);assert.match(el('reminder-notes').children[0].textContent,/本地文字提醒/);
fixture.camera={ready:true,vision_paused:true,privacy_active:true};fixture.totals.privacy=90;fixture.intervals=[{kind:'person',state:'privacy',start,end:start+90}];await vm.runInContext('refresh()',context);assert.match(el('current').textContent,/隐私保护中/);assert.equal(el('show').disabled,true);assert.equal(el('frame').hidden,true);assert.equal(el('live-canvas').hidden,true);assert.match(el('live-message').textContent,/隐私保护/);
fixture.camera={ready:true,vision_paused:false,person:'sit',presence:'present'};await vm.runInContext('refresh()',context);assert.equal(el('show').disabled,false);assert.match(html,/setInterval\([\s\S]*15000/);const segments = [
  {kind:'person',state:'sit',start:10,end:15},
  {kind:'person',state:'sit',source:'estimated',start:15,end:24},
  {kind:'presence',state:'present',start:10,end:24},
  {kind:'person',state:'sit',start:26,end:32}
];
context.segments=segments;
assert.equal(vm.runInContext("postureRuns(segments).length",context),2);
assert.equal(vm.runInContext("postureRuns(segments)[0].end",context),24);
assert.equal(vm.runInContext("estimatedSeconds(segments,'sit')",context),9);
assert.match(vm.runInContext("durationHTML(19)",context),/19.*秒/);
fixture.camera={ready:true,person:'unknown',presence:'present',pose_source:'multiple_people'};
fixture.totals.stand=3599;
await vm.runInContext('refresh()',context);
assert.equal(el('goal-percent').textContent,'99%');
assert.match(el('current').textContent,/多人在位/);
assert.match(el('hour-chart').children.find(n=>n['aria-label'])['aria-label'],/无记录 1 小时/);
fixture.date=fixture.today='2026-11-12';
await vm.runInContext('refresh()',context);
assert.equal(el('day').value,'2026-11-12');
assert.equal(requests.filter(r=>r.path.startsWith('/api/dashboard')).at(-1).path,'/api/dashboard');
fixture.date='2026-11-11';
el('previous').onclick(); await new Promise(setImmediate);
assert.equal(requests.filter(r=>r.path.startsWith('/api/dashboard')).at(-1).path,'/api/dashboard?date=2026-11-11');
const workingFetch=context.fetch;
context.fetch=async()=>{throw Error('disconnected')};
await vm.runInContext('refresh()',context);
assert.equal(el('connection').textContent,'连接中断');assert.equal(el('live-canvas').hidden,true);
context.fetch=workingFetch;fixture.date=fixture.today;
el('today').onclick(); await new Promise(setImmediate);
assert.equal(el('day').value,'2026-11-12');assert.match(el('connection').textContent,/已连接/);
assert.equal(el('focus-form').hidden,false);
replies['/api/focus'] = {active:{id:'active', title:'写初稿', minutes:25, due_at:start+1500},
                         recent:[], server_now:start};
await vm.runInContext('syncFocus()',context);
assert.equal(el('focus-name').textContent,'写初稿');
assert.equal(el('focus-clock').textContent,'25:00');
assert.equal(el('focus-form').hidden,true);
const taskBox = () => el('tasks').children[0].children[0].children[0];
assert.equal(taskBox().dataset.taskId, todo.id);
assert.equal(taskBox().checked, false);
replies['/api/todos'] = {items:[], next_cursor:null};
await taskBox().onchange();
await new Promise(setImmediate);
assert.deepEqual(requests.find(r=>r.path==='/api/todos/completion').body, {id:todo.id, completed:true});
assert.match(el('tasks').children[0].textContent,/没有待处理/);
replies['/api/todos'] = {items:[{...todo, completed:true}], next_cursor:null};
inboxTabs[1].onclick();
await new Promise(setImmediate);
assert.equal(taskBox().checked, true);
assert.ok(requests.some(r=>r.path==='/api/todos?completed=true'));
console.log('Dashboard focus, todo, voice, privacy, continuity, goal, date rollover, disconnect and recovery PASS')})().catch(e=>{console.error(e);process.exitCode=1});
