const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const source=fs.readFileSync(require('node:path').join(__dirname,'..','index.html'),'utf8');
function section(start,end){return source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));}
function context(extra={}){
  const c=vm.createContext({Promise,String,encodeURIComponent,DCC_SHEET_ID:'dcc-sheet',sheetId:'new-sheet',accessToken:'test',...extra});
  vm.runInContext(section('  var ANIMALS_HEADERS =','  window.createWorkspace ='),c);
  vm.runInContext(section('  var animalRecordsCache =','  function loadTab('),c);
  vm.runInContext(section('  function loadAnimalRecords()','  function lookupAnimal('),c);
  return c;
}
test('new workspaces contain one animal table and a compatible scan log',()=>{
  const c=context();
  const tabs=c.workspaceDefinition('Test').sheets.map(s=>s.properties.title);
  assert.deepEqual(Array.from(tabs),['Animals','HeadCounter']);
  const headers=c.workspaceHeaders().data;
  assert.equal(headers[0].values[0].length,16);
  assert.equal(headers[1].values[0].length,13);
  assert.equal(headers[1].range,'HeadCounter!A1:M1');
  assert.equal(headers[1].values[0][3],'Tag EPC');
});
test('Animals wins over split trackers, except established DCC',()=>{
  const c=context();
  assert.equal(c.animalSchema(['Animals','Ranch Tracker','Calf Tracker'],'new'),'animals');
  assert.equal(c.animalSchema(['Animals','Ranch Tracker','Calf Tracker'],'dcc-sheet'),'legacy');
  assert.equal(c.animalSchema(['Ranch Tracker','Calf Tracker'],'old'),'legacy');
  assert.throws(()=>c.animalSchema(['HeadCounter'],'new'),/Animals table/);
});
test('species-neutral records preserve identities, leading zeros and parent links',()=>{
  const c=context();
  const rows=c.animalsFromRows([{'Tag EPC':'740086010000674','Animal ID':'AN-1','Printed Tag':'0674',Name:'Test cougar',Species:'Cougar','Description':'Test description','Dam ID':'AN-2','Sire ID':'AN-3'},{}]);
  assert.equal(rows.length,1);
  assert.equal(rows[0].tag,'0674');assert.equal(rows[0].uhf,'740086010000674');
  assert.equal(rows[0].species,'Cougar');assert.equal(rows[0].dam,'AN-2');
  assert.equal(rows[0].sire,'AN-3');assert.equal(rows[0].description,'Test description');
});
test('older Animals header aliases are readable without split trackers',()=>{
  const c=context();const r=c.animalsFromRows([{'Tag EPC':'123456789012345','Animal ID':'0001','Dam Tag':'0002','Sire Tag':'0003'}])[0];
  assert.equal(r.tag,'0001');assert.equal(r.dam,'0002');assert.equal(r.sire,'0003');
});
test('empty Animals table is valid and never reads legacy trackers',async()=>{
  const reads=[];
  const c=context({fetchWithAuthRetry:async()=>({ok:true,json:async()=>({sheets:[{properties:{title:'Animals'}}]})}),loadTab:async name=>{reads.push(name);return [];},loadRanchTracker:()=>{throw Error('legacy read');}});
  const records=await c.loadAnimalRecords();assert.equal(records.length,0);assert.deepEqual(reads,['Animals']);
});
test('concurrent scans share one lookup and cached results',async()=>{
  let requests=0,loads=0;
  const c=context({fetchWithAuthRetry:async()=>{requests++;return {ok:true,json:async()=>({sheets:[{properties:{title:'Animals'}}]})};},loadTab:async()=>{loads++;return [{'Tag EPC':'123456789012345','Species':'Goat'}];}});
  const [a,b]=await Promise.all([c.loadAnimalRecords(),c.loadAnimalRecords()]);
  assert.equal(a,b);assert.equal(requests,1);assert.equal(loads,1);
  assert.equal(await c.loadAnimalRecords(),a);assert.equal(requests,1);
});
test('permission error stays visible; later lookup retries without a legacy fallback',async()=>{
  let requests=0;
  const c=context({fetchWithAuthRetry:async()=>({ok:++requests>1,status:403,json:async()=>({sheets:[{properties:{title:'Animals'}}]})}),loadTab:async()=>[],loadRanchTracker:()=>{throw Error('legacy fallback');}});
  await assert.rejects(c.loadAnimalRecords(),/HTTP 403/);
  await c.loadAnimalRecords();assert.equal(requests,2);
});
test('existing split workbook uses both established loaders',async()=>{
  const reads=[];
  const c=context({fetchWithAuthRetry:async()=>({ok:true,json:async()=>({sheets:['Calf Tracker','Ranch Tracker'].map(title=>({properties:{title}}))})}),loadTab:async name=>{reads.push(name);return [{'Calf Tag':'0001','UHF#':'123456789012345'}];},loadRanchTracker:async()=>[{source:'Ranch Tracker',tag:'0002'}]});
  const r=await c.loadAnimalRecords();assert.equal(r.length,2);assert.equal(r[0].tag,'0001');assert.deepEqual(reads,['Calf Tracker']);
});
test('workspace change cannot cache another operation’s records',async()=>{
  let release;
  const c=context({fetchWithAuthRetry:async()=>({ok:true,json:async()=>({sheets:[{properties:{title:'Animals'}}]})}),loadTab:()=>new Promise(resolve=>{release=resolve;})});
  const result=c.loadAnimalRecords();await new Promise(setImmediate);
  c.sheetId='another-sheet';release([{'Tag EPC':'123456789012345'}]);
  await assert.rejects(result,/Operation changed/);assert.equal(c.animalRecordsCache,null);
});
test('public front door preserves scanner host and OAuth code callbacks',()=>{
  const script=source.match(/<script>([\s\S]*?)<\/script>/)[1];
  for(const [hostname,pathname,search,redirect] of [['herdmate.ag','/','',true],['scanner.herdmate.ag','/','',false],['herdmate.ag','/','?code=test',false],['herdmate.ag','/gateway.html','',false]]){
    let changed=false;vm.runInNewContext(script,{URLSearchParams,location:{hostname,pathname,search,replace:()=>{changed=true;}}});assert.equal(changed,redirect);
  }
});
