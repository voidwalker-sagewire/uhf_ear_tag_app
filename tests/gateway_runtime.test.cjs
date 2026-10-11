const {test} = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const root = path.join(__dirname, '..');
const source = fs.readFileSync(path.join(root, 'index.html'), 'utf8');
function between(start, end) { return source.slice(source.indexOf(start), source.indexOf(end, source.indexOf(start))); }
function context(overrides = {}) {
  const storage = new Map();
  return vm.createContext({URLSearchParams, window:{location:{pathname:'/gateway.html',search:''}},
    localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
    operationId:'stale',sheetId:'stale-sheet',operationName:'Stale',operationFeatures:{},
    gatewayTenantMode:true,accessToken:'fake-token',signedInEmail:'test@example.com',
    SCOUT_CONFIG_API:'https://example.invalid/config',
    debugLog(){}, showScreen(){}, showToast(){}, animalRecordsCache:null,
    saveLocal(id,name){this.sheetId=id;this.operationName=name;}, launchApp(){},
    ...overrides});
}
test('cleared-state dedicated page always selects Gateway',()=>{
  const c=context();
  vm.runInContext(between('  var gatewayTenantMode =','  var operationId ='),c);
  assert.equal(c.gatewayTenantMode,true);
  assert.equal(c.localStorage.getItem('hm_tenant_mode'),'gateway');
  c.window.location={pathname:'/',search:'?code=fake'};
  vm.runInContext(between('  var gatewayTenantMode =','  var operationId ='),c);
  assert.equal(c.gatewayTenantMode,true);
});
test('legacy root with no route marker retains the working DCC entry',()=>{
  const c=context({window:{location:{pathname:'/',search:''}}});
  vm.runInContext(between('  var gatewayTenantMode =','  var operationId ='),c);
  assert.equal(c.gatewayTenantMode,false);
});
for (const operation of [{id:'a',name:'Ranch A',sheet_id:'sheet-a'},{id:'b',name:'Ranch B',sheet_id:'sheet-b'}]) {
  test(`Gateway resolves only returned operation ${operation.id}`,async()=>{
    let launched=false, saved;
    const c=context({saveLocal:(id,name)=>{saved={id,name};},launchApp:()=>{launched=true;},
      fetchWithAuthRetry:async()=>({ok:true,status:200,json:async()=>({status:'ACTIVE',operation,features:{field_head_count:true}})})});
    vm.runInContext(between('  function resolveGatewayTenant()', '  function launchApp()'),c);
    vm.runInContext('resolveGatewayTenant()',c);
    await new Promise(setImmediate);
    assert.equal(c.operationId,operation.id);
    assert.deepEqual(saved,{id:operation.sheet_id,name:operation.name});
    assert.equal(launched,true);
  });
}
for(const payload of [{status:'ONBOARDING_REQUIRED'},{status:'ACCESS_DISABLED'},{status:'ACTIVE',operation:{id:'a',name:'A'},features:{field_head_count:true}},
  {status:'ACTIVE',operation:{id:'a',name:'A',sheet_id:'a'},features:{field_head_count:false}}]) {
  test(`blocked configuration ${JSON.stringify(payload)} cannot launch DCC`,async()=>{
    let launched=false;
    const c=context({launchApp:()=>{launched=true;},fetchWithAuthRetry:async()=>({ok:true,status:200,json:async()=>payload})});
    vm.runInContext(between('  function resolveGatewayTenant()', '  function launchApp()'),c);
    vm.runInContext('resolveGatewayTenant()',c);
    await new Promise(setImmediate);
    assert.equal(launched,false);assert.equal(c.sheetId,'');assert.equal(c.operationId,'');
  });
}
test('Gateway authentication cannot invoke the legacy DCC save',()=>{
  let resolved=false;
  const c=context({resolveGatewayTenant:()=>{resolved=true;},saveLocal:()=>{throw Error('legacy fallback');}});
  vm.runInContext(between('  function onSignedIn()', '  function resolveGatewayTenant()'),c);
  vm.runInContext('onSignedIn()',c);assert.equal(resolved,true);
});
test('dedicated page matches the scanner source with a deployment marker',()=>{
  assert.equal(fs.readFileSync(path.join(root,'gateway.html'),'utf8'),source.replace('<head>','<head>\n<meta name="reaper-scout-route" content="gateway-rs-tx-007">'));
});
for (const owner of ['a', null]) {
  test(`operation b cannot upload queued ${owner || 'legacy'} scans`,()=>{
    let requests=0;
    const queue=[{operationId:owner,record:{tag:'fake'},mode:'Head Count'}];
    const c=context({operationId:'b',sheetId:'sheet-b',queueLoad:()=>queue,
      setSheetsBar(){},fetchWithAuthRetry(){requests++;throw Error('cross-tenant write');}});
    vm.runInContext(between('  function flushOfflineQueue()', '  // ── CHROME CHECK'),c);
    vm.runInContext('flushOfflineQueue()',c);
    assert.equal(requests,0);assert.equal(queue.length,1);
  });
}
