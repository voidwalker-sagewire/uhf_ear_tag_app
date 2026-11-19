const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const {scanner,section}=require('./helpers/scanner.cjs');
function offline(extra={}){return scanner({accessToken:null,...extra});}
function tags(n){return Array.from({length:n},(_,i)=>'74008601'+String(i+100).padStart(7,'0'));}
function runTimer(c,ms){
  const t=c.timers.filter(v=>v.ms===ms);
  assert.ok(t.length>0,'expected scheduled timer '+ms);
  t[0].fn();
}
test('4528 accepted reads, 86 EPCs: UI and repeated writes are batched',()=>{
  let renders=0,stats=0;
  const c=offline({renderTable(){renders++;},updateStats(){stats++;}});
  const epcs=tags(86);
  for(let i=0;i<4528;i++)c.processHCTag(epcs[i%86]);
  assert.equal(Object.keys(c.uniqueTags).length,86);
  assert.equal(Object.values(c.uniqueTags).reduce((x,r)=>x+r.readCount,0),4528);
  assert.equal(renders,0,'no full-table rebuild should run in each callback');
  assert.equal(c.queueLoad().length,86,'new EPCs are durable immediately');
  c.flushPendingReadUpdates();
  const q=c.queueLoad();
  assert.equal(q.length,86);
  assert.equal(q.reduce((n,x)=>n+Number(x.record.readCount||0),0),4528);
  runTimer(c,500);
  assert.equal(renders,1);assert.equal(stats,1);
});
test('Stop Capture ignores office EPCs, preserves saved data, then resumes',()=>{
  const elements={};
  function get(id){return elements[id]||(elements[id]={disabled:false,value:'',textContent:'',className:'',focus(){}});}
  const c=offline({accessToken:'token',sheetId:'test',operationId:'dcc',document:{getElementById:get}});
  vm.runInContext(section('  // ── PROCESS INCOMING TAG', '  // SETTINGS'),c);
  c.currentMode='hc';c.lastScanTime={};c.flushOfflineQueue=()=>Promise.resolve();
  c.processIncomingTag('740086010000141');
  assert.equal(Object.keys(c.uniqueTags).length,1);
  c.window.toggleHeadCountCapture();
  assert.equal(c.headCountPaused,true);
  c.processIncomingTag('740086010000311');
  assert.equal(Object.keys(c.uniqueTags).length,1);
  assert.equal(get('tagInput').disabled,false);
  assert.equal(get('addBtn').disabled,true);
  c.window.toggleHeadCountCapture();
  c.processIncomingTag('740086010000311');
  assert.equal(Object.keys(c.uniqueTags).length,2);
});
test('new tag is durably queued while offline',()=>{
  const c=offline();
  c.processHCTag('740086010000141');
  assert.equal(c.queueLoad().length,1);
  assert.equal(c.queueLoad()[0].record.tag,'740086010000141');
});
test('repeat-count persistence stays pending until flush; unique remains durable',()=>{
  const c=offline();
  c.processHCTag('740086010000141');
  c.processHCTag('740086010000141');
  assert.equal(c.uniqueTags['740086010000141__Barn'].readCount,2);
  assert.equal(c.queueLoad()[0].record.readCount,1);
  c.flushPendingReadUpdates();
  assert.equal(c.queueLoad()[0].record.readCount,2);
});
test('dense repeat stream defers queue rewrites for five seconds',()=>{
  const c=offline();
  c.processHCTag('740086010000141');
  for(let i=0;i<1000;i++)c.processHCTag('740086010000141');
  assert.equal(c.queueLoad()[0].record.readCount,1);
  assert.equal(c.timers.filter(v=>v.ms===5000).length,1);
  c.flushPendingReadUpdates();
  assert.equal(c.queueLoad()[0].record.readCount,1001);
});
test('head-count journal restores tags only for the verified destination',()=>{
  const c=offline();
  c.processHCTag('740086010000141');
  c.currentPasture='North Lot';
  c.saveHeadCountSession();
  c.uniqueTags={};c.currentPasture='';c.sessionId='new-session';
  assert.equal(c.restoreHeadCountSession(),true);
  assert.equal(Object.keys(c.uniqueTags).length,1);
  assert.equal(c.currentPasture,'North Lot');
  assert.equal(c.sessionId,'session-original');
  c.sheetId='other-sheet';c.uniqueTags={};
  assert.equal(c.restoreHeadCountSession(),false);
  assert.equal(Object.keys(c.uniqueTags).length,0);
});
