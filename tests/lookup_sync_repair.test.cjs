const {test}=require('node:test');
const assert=require('node:assert/strict');
const {scanner,tick}=require('./helpers/scanner.cjs');
function response(data,status=200){return {ok:status>=200&&status<300,status,json:async()=>JSON.parse(JSON.stringify(data))};}
function sheets(){
  const rows=[],requests=[];
  async function fetch(url,options={}){
    const method=options.method||'GET';requests.push({url,method,body:options.body});
    const range=decodeURIComponent(url.split('/values/')[1]||'').split('?')[0];
    if(method==='POST'){
      rows.push(JSON.parse(options.body).values[0]);
      return response({updates:{updatedRange:"HeadCounter!A"+(rows.length+1)+":M"+(rows.length+1)}});
    }
    if(method==='PUT'){
      const n=Number(range.match(/!J(\d+)/)[1]);
      rows[n-2][9]=JSON.parse(options.body).values[0][0];return response({});
    }
    if(range.endsWith('!A:A'))return response({values:[['HCID'],...rows.map(r=>[r[0]])]});
    const n=Number(range.match(/!A(\d+)/)[1]);return response({values:rows[n-2]?[[rows[n-2][0]]]:[]});
  }
  return {rows,requests,fetch};
}
function observation(tag='740086010000014'){return {tag,pasture:'Barn',firstDate:'10/22/2026',firstTime:'08:00:00 AM',readCount:1};}
function ready(c){const q=c.queueLoad();q.forEach(x=>x.notBefore=0);c.queueSave(q);}
test('overlapping flush calls send two different queued observations once each',async()=>{
  const api=sheets();let release,calls=0;
  const c=scanner({accessToken:null,fetchWithAuthRetry:async(...args)=>{
    if(++calls===1)await new Promise(r=>release=r);return api.fetch(...args);
  }});
  c.queueAdd(observation('A'),'Head Count');c.queueAdd(observation('B'),'Head Count');
  c.accessToken='token';const first=c.flushOfflineQueue(),second=c.flushOfflineQueue();await tick();
  assert.equal(first,second);assert.equal(calls,1);release();await first;
  assert.deepEqual(api.rows.map(r=>r[3]),['A','B']);assert.equal(c.queueLoad().length,0);
});
test('an append committed before its response was lost reconciles instead of duplicating',async()=>{
  const api=sheets();let lose=true;
  const c=scanner({fetchWithAuthRetry:async(url,o)=>{
    const r=await api.fetch(url,o);
    if(o?.method==='POST'&&lose){lose=false;throw Error('Failed to fetch');}
    return r;
  }});
  const r=observation();await c.writeToSheet(r,'Head Count');
  assert.equal(api.rows.length,1);assert.equal(c.queueLoad().length,1);
  const stable=c.queueLoad()[0].id;
  await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][0],stable);
  assert.equal(c.queueLoad().length,0);
});
test('ambiguous append survives page restart with original session and author',async()=>{
  const api=sheets();
  const c=scanner({fetchWithAuthRetry:async(url,o)=>{await api.fetch(url,o);throw Error('lost');}});
  await c.writeToSheet(observation(),'Head Count');
  const saved=c.localStorage.getItem(c.QUEUE_KEY);
  const restart=scanner({sessionId:'new-session',signedInEmail:'new-login@example.com',fetchWithAuthRetry:api.fetch});
  restart.localStorage.setItem(restart.QUEUE_KEY,saved);
  await restart.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][1],'session-original');
  assert.equal(api.rows[0][12],'owner@example.com');assert.equal(restart.queueLoad().length,0);
});
test('reads arriving during append are saved by updating the same row',async()=>{
  const api=sheets();let release,first=true;
  const c=scanner({fetchWithAuthRetry:async(url,o)=>{
    if(o?.method==='POST'&&first){first=false;await new Promise(r=>release=r);}
    return api.fetch(url,o);
  }});
  const r=observation();c.uniqueTags[r.tag+'__Barn']=r;
  const pending=c.writeToSheet(r,'Head Count');await tick();
  r.readCount=8;c.writeToSheet(r,'Head Count');release();await pending;
  assert.equal(api.rows.length,1);assert.equal(c.queueLoad().length,1);
  ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][9],8);
  assert.equal(r.synced,true);assert.equal(c.queueLoad().length,0);
});
test('repeated reads after a successful upload update J instead of appending',async()=>{
  const api=sheets(),c=scanner({fetchWithAuthRetry:api.fetch});const r=observation();
  await c.writeToSheet(r,'Head Count');r.readCount=3;await c.writeToSheet(r,'Head Count');
  ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][9],3);
});
test('Google success without an updatedRange still reconciles the next count by ID',async()=>{
  const api=sheets();const c=scanner({fetchWithAuthRetry:async(url,o)=>{
    const r=await api.fetch(url,o);return o?.method==='POST'?response({}):r;
  }});
  const r=observation();await c.writeToSheet(r,'Head Count');
  r.readCount=4;await c.writeToSheet(r,'Head Count');ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][9],4);
});
test('row movement is detected before a count update touches another animal',async()=>{
  const api=sheets(),c=scanner({fetchWithAuthRetry:api.fetch}),r=observation();
  await c.writeToSheet(r,'Head Count');
  api.rows.unshift(['unrelated','','','other','','','','','',99]);
  r.readCount=5;await c.writeToSheet(r,'Head Count');ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows[0][9],99);assert.equal(api.rows[1][9],5);
});
test('duplicate IDs are held for review rather than choosing or adding another row',async()=>{
  const api=sheets(),c=scanner({fetchWithAuthRetry:api.fetch}),r=observation();
  await c.writeToSheet(r,'Head Count');api.rows.push([...api.rows[0]]);
  r.sheetRow=null;r.readCount=2;await c.writeToSheet(r,'Head Count');
  // Existing synced state made this a reconciliation, even with no row hint.
  ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows.length,2);assert.equal(c.queueLoad().length,1);
});
test('reconciliation failure cannot fall through to a blind append',async()=>{
  const api=sheets();const c=scanner({fetchWithAuthRetry:async(url,o)=>{
    if(!o?.method)return response({},403);return api.fetch(url,o);
  }});
  const r=observation();await c.writeToSheet(r,'Head Count');
  r.readCount=2;await c.writeToSheet(r,'Head Count');ready(c);await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][9],1);assert.equal(c.queueLoad().length,1);
});
test('stale upload cannot acknowledge or upload under the next operation',async()=>{
  const api=sheets();let release;
  const c=scanner({fetchWithAuthRetry:async(...args)=>{await new Promise(r=>release=r);return api.fetch(...args);}});
  const pending=c.writeToSheet(observation(),'Head Count');await tick();
  c.onboardingAccountEpoch++;c.operationId='demo';c.sheetId='demo-sheet';c.signedInEmail='business@example.com';
  release();await pending;
  assert.equal(c.queueLoad().length,1);assert.equal(c.queueLoad()[0].operationId,'dcc');
  await c.flushOfflineQueue();
  assert.equal(api.requests.length,1);assert.ok(api.requests[0].url.includes('/dcc-sheet/'));
});
test('queue removal acknowledges an exact ID rather than the next queue index',()=>{
  const c=scanner({accessToken:null});
  const a=observation('A'),b=observation('B');c.queueAdd(a,'Head Count');c.queueAdd(b,'Head Count');
  c.queueRemove(a.syncId,'dcc',1);c.queueRemove(a.syncId,'dcc',1);
  assert.equal(c.queueLoad().length,1);assert.equal(c.queueLoad()[0].record.tag,'B');
});
test('old owned entries migrate durably without inventing original author or session',async()=>{
  const api=sheets(),c=scanner({fetchWithAuthRetry:api.fetch});
  c.queueSave([{operationId:'dcc',mode:'Head Count',record:observation()}]);
  await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][1],'');assert.equal(api.rows[0][12],'');
});
test('storage failure prevents network writes and reports a local-save error',async()=>{
  let calls=0,message='';
  const c=scanner({fetchWithAuthRetry:async()=>{calls++;},setSheetsBar:(_s,m)=>message=m});
  c.localStorage.setItem=()=>{throw Error('QuotaExceeded');};
  await c.writeToSheet(observation(),'Head Count');
  assert.equal(calls,0);assert.match(message,/Local storage/);
});
test('serialized Beacon recovery preserves its original target and operation',async()=>{
  const api=sheets(),c=scanner({bbTarget:'target-original',fetchWithAuthRetry:api.fetch});
  const r={tag:'beacon',date:'10/22/2026',time:'08:00 AM',isTarget:true};
  await c.writeToSheet(r,'Bovine Beacon');
  assert.equal(api.rows[0][4],'target-original');assert.equal(api.rows[0][9],'DCC');
});

function lookupScanner(extra={}){
  const c=scanner(extra);
  c.writeToSheet=()=>Promise.resolve();
  c.loadTab=async()=>[{'Calf Tag':'917','UHF#':'740086010000014'}];
  c.loadRanchTracker=async()=>[{source:'Ranch Tracker',tag:'933',uhf:'740086010000015'}];
  return c;
}
test('same-tag rescan recovers a failed 0014 lookup to animal 917',async()=>{
  let fail=true;
  const c=lookupScanner({fetchWithAuthRetry:async()=>fail?response({},403):response({sheets:[]})});
  c.processHCTag('740086010000014');await tick();
  const r=c.uniqueTags['740086010000014__Barn'];assert.ok(r.animalLookupError);
  fail=false;r.animalLookupRetryAt=0;
  c.processHCTag('740086010000014');await tick();
  assert.equal(r.animal.tag,'917');assert.equal(r.animalLookupError,null);assert.equal(r.readCount,2);
});
test('later successful 0015 lookup also repairs the earlier 0014 card',async()=>{
  let fail=true;
  const c=lookupScanner({fetchWithAuthRetry:async()=>fail?response({},403):response({sheets:[]})});
  c.processHCTag('740086010000014');await tick();fail=false;
  c.processHCTag('740086010000015');await tick();
  assert.equal(c.uniqueTags['740086010000014__Barn'].animal.tag,'917');
  assert.equal(c.uniqueTags['740086010000015__Barn'].animal.tag,'933');
});
test('failed source never becomes an unlinked result; bounded retry can recover it',async()=>{
  const c=lookupScanner({fetchWithAuthRetry:async()=>response({sheets:[]})});
  let fail=true;c.loadRanchTracker=async()=>{if(fail){const e=Error('Ranch unavailable');e.status=503;throw e;}return [];};
  c.processHCTag('740086010000014');await tick();
  const r=c.uniqueTags['740086010000014__Barn'];assert.ok(r.animalLookupError);
  assert.notEqual(r.animalLookupError,null);fail=false;r.animalLookupRetryAt=0;
  c.retryFailedLookups();await tick();assert.equal(r.animal.tag,'917');
});
test('only a completed full lookup labels 0017 unlinked',async()=>{
  const c=lookupScanner({fetchWithAuthRetry:async()=>response({sheets:[]})});
  c.processHCTag('740086010000017');await tick();
  const r=c.uniqueTags['740086010000017__Barn'];assert.equal(r.animal,null);assert.equal(r.animalLookupError,null);
});
test('automatic lookup attempts stop at three and concurrent rescans share the request',async()=>{
  const c=lookupScanner({fetchWithAuthRetry:async()=>response({sheets:[]})});
  let release,loads=0;c.loadTab=()=>{loads++;return new Promise((_resolve,reject)=>release=reject);};
  c.processHCTag('740086010000014');c.processHCTag('740086010000014');await tick();
  assert.equal(loads,1);
  release(Error('network'));await tick();
  const r=c.uniqueTags['740086010000014__Barn'];
  for(let i=0;i<2;i++){r.animalLookupRetryAt=0;c.retryFailedLookups();await tick();release(Error('network'));await tick();}
  r.animalLookupRetryAt=0;c.retryFailedLookups();await tick();
  assert.equal(loads,3);
});

test('reconnection restarts bounded lookups after earlier attempts were exhausted',async()=>{
  const c=lookupScanner({fetchWithAuthRetry:async()=>response({sheets:[]})});
  const r=observation();r.animal=null;r.animalLookupError='network';r.animalLookupRetryable=true;
  r.animalLookupAttempts=3;r.animalLookupRetryAt=Date.now()+100000;
  c.uniqueTags[r.tag+'__Barn']=r;
  c.events.online();await tick();
  assert.equal(r.animal.tag,'917');assert.equal(r.animalLookupError,null);
});
test('malformed durable queue is preserved and cannot be treated as empty/synced',async()=>{
  let calls=0,message='';
  const c=scanner({fetchWithAuthRetry:async()=>{calls++;},setSheetsBar:(_s,m)=>message=m});
  c.localStorage.setItem(c.QUEUE_KEY,'broken JSON');
  await c.flushOfflineQueue();
  assert.equal(calls,0);assert.equal(c.localStorage.getItem(c.QUEUE_KEY),'broken JSON');
  assert.match(message,/incomplete/);assert.equal(c.queueCountForStatus(),'unreadable');
});
test('real DCC loaders retrieve 917 from Calf Tracker and 933 from Ranch Tracker',async()=>{
  const c=scanner({fetchWithAuthRetry:async(url)=>{
    if(url.includes('?fields='))return response({sheets:['Animals','Calf Tracker','Ranch Tracker'].map(title=>({properties:{title}}))});
    if(url.includes('values:batchGet'))return response({valueRanges:[
      {values:[['ID','DisplayID','Tag #'],['fixture','933_fixture','933']]},
      {values:[['Pasture','Herd','Breed','Sex','Type','Birth Date','Color'],['Barn','','Akaushi','Male','Bull','3/31/2025','Red']]},
      {values:[['Status'],['Active']]},{values:[['Notes'],['']]},
      {values:[['UHF#'],['740086010000015']]}
    ]});
    assert.ok(decodeURIComponent(url).includes('Calf Tracker!A:BZ'));
    return response({values:[['Calf Tag','UHF#','Status'],['917','740086010000014','Active']]});
  }});
  c.writeToSheet=()=>Promise.resolve();
  c.processHCTag('740086010000014');await tick();
  c.processHCTag('740086010000015');await tick();
  assert.equal(c.uniqueTags['740086010000014__Barn'].animal.tag,'917');
  assert.equal(c.uniqueTags['740086010000015__Barn'].animal.tag,'933');
});
test('all inline scripts parse in both generated entry points',()=>{
  const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
  for(const file of ['index.html','gateway.html']){
    const html=fs.readFileSync(path.join(__dirname,'..',file),'utf8');
    for(const match of html.matchAll(/<script\b[^>]*>([\s\S]*?)<\/script>/g))new vm.Script(match[1],{filename:file});
  }
});

test('lookup response after account switch cannot alter an old card',async()=>{
  let release;
  const c=lookupScanner({fetchWithAuthRetry:()=>new Promise(r=>release=r)});
  c.processHCTag('740086010000014');await tick();
  const old=c.uniqueTags['740086010000014__Barn'];c.onboardingAccountEpoch++;c.sheetId='demo-sheet';c.uniqueTags={};
  release(response({sheets:[]}));await tick();
  assert.equal(old.animal,undefined);assert.equal(c.animalRecordsCache,null);
});
test('transient animal GET retries once with stage diagnostics, while 403 does not retry',async()=>{
  const stages=[];let calls=0;
  const c=scanner({setTimeout:fn=>{setImmediate(fn);},debugLog:(event,data)=>stages.push({event,data}),
    fetchWithAuthRetry:async()=>{calls++;return response({},calls===1?503:200);}});
  await c.fetchAnimalData('https://example.invalid',{},'Calf Tracker');assert.equal(calls,2);
  assert.equal(stages[0].data.stage,'Calf Tracker');
  calls=0;c.fetchWithAuthRetry=async()=>{calls++;return response({},403);};
  await assert.rejects(c.fetchAnimalData('https://example.invalid',{},'Calf Tracker'),/HTTP 403/);
  assert.equal(calls,1);
});

test('a network failure after successful token refresh is not relabeled authentication failure',async()=>{
  const vm=require('node:vm'),{section}=require('./helpers/scanner.cjs');let calls=0;
  const c=vm.createContext({accessToken:'fake',onboardingAccountEpoch:0,debugLog(){},refreshAccessToken:async()=>{},
    fetch:async()=>{if(++calls===1)return {status:401};throw Error('Failed to fetch');}});
  vm.runInContext(section('  function fetchWithAuthRetry(', '  function silentRefresh('),c);
  await assert.rejects(c.fetchWithAuthRetry('https://example.invalid',{},'lookup'),e=>e.message==='Failed to fetch' && !e.status && !e.authRefreshFailed);
});
test('supporting tabs share a Web Lock so they cannot drain the same head concurrently',async()=>{
  const api=sheets();let release,calls=0,tail=Promise.resolve(),locks=0;
  const sharedLock={request(_name,fn){locks++;const p=tail.then(fn);tail=p.catch(()=>{});return p;}};
  const fetch=async(...args)=>{if(++calls===1)await new Promise(r=>release=r);return api.fetch(...args);};
  const a=scanner({accessToken:null,navigator:{locks:sharedLock},fetchWithAuthRetry:fetch});
  a.queueAdd(observation('A'),'Head Count');a.queueAdd(observation('B'),'Head Count');
  const b=scanner({navigator:{locks:sharedLock},fetchWithAuthRetry:fetch,localStorage:a.localStorage});
  a.accessToken='token';const first=a.flushOfflineQueue(),second=b.flushOfflineQueue();await tick();
  assert.equal(calls,1);release();await Promise.all([first,second]);
  assert.deepEqual(api.rows.map(r=>r[3]),['A','B']);assert.equal(locks,2);
});
test('a failed count PUT retries as an update of the original row',async()=>{
  const api=sheets();let fail=false;
  const c=scanner({fetchWithAuthRetry:async(url,o)=>{
    if(o?.method==='PUT'&&fail){fail=false;throw Error('network');}return api.fetch(url,o);
  }});
  const r=observation();await c.writeToSheet(r,'Head Count');
  fail=true;r.readCount=9;await c.writeToSheet(r,'Head Count');ready(c);await c.flushOfflineQueue();
  assert.equal(c.queueLoad().length,1);
  await c.flushOfflineQueue();
  assert.equal(api.rows.length,1);assert.equal(api.rows[0][9],9);assert.equal(c.queueLoad().length,0);
});
test('unknown queue owners stay untouched and cannot be migrated into the current operation',async()=>{
  let calls=0;const c=scanner({fetchWithAuthRetry:async()=>{calls++;}});
  const original=[{mode:'Head Count',record:observation()}];c.queueSave(original);
  await c.flushOfflineQueue();
  assert.equal(calls,0);assert.equal(JSON.stringify(c.queueLoad()),JSON.stringify(original));
});


