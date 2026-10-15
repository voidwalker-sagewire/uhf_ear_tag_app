const {test}=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');
const vm=require('node:vm');
const crypto=require('node:crypto').webcrypto;
const source=fs.readFileSync(require('node:path').join(__dirname,'..','index.html'),'utf8');
function section(start,end){return source.slice(source.indexOf(start),source.indexOf(end,source.indexOf(start)));}
function context(extra={}){
  const elements={};const storage=new Map();
  const c=vm.createContext({Promise,URLSearchParams,TextEncoder,Uint8Array,btoa,Array,Date,
    window:{crypto,location:{search:'',href:''},history:{replaceState(){}}},
    sessionStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    localStorage:{getItem:k=>storage.get(k),setItem:(k,v)=>storage.set(k,v),removeItem:k=>storage.delete(k)},
    document:{title:'Test',getElementById:id=>elements[id]||(elements[id]={value:'My Ranch',style:{}})},
    REAPER_ONBOARDING_API:'https://example.invalid/onboarding',SCOUT_CONFIG_API:'https://example.invalid/config',
    CLIENT_ID:'test-client',SCOPES:'openid test-scope',AUTH_SERVER:'https://auth.invalid',
    onboardingBusy:false,onboardingAccountEpoch:0,gatewayTenantMode:true,
    operationId:'stale',sheetId:'stale',operationName:'Stale',operationFeatures:{},
    animalRecordsCache:null,animalRecordsPromise:null,accessToken:'test',signedInEmail:'owner@example.com',
    setStep(){},showScreen(){},showToast(){},debugLog(){},saveLocal(){},launchApp(){},
    ...extra});
  vm.runInContext(section('  function showGatewayOnboarding(', '  window.createWorkspace ='),c);
  return c;
}
test('unknown account enters setup without a connected Sheet or scanning',async()=>{
  let screen,launched=false;
  const c=context({showScreen:s=>screen=s,launchApp:()=>launched=true,
    fetchWithAuthRetry:async()=>({ok:true,status:200,json:async()=>({status:'ONBOARDING_REQUIRED'})})});
  vm.runInContext(section('  function resolveGatewayTenant()', '  function launchApp()'),c);
  await c.resolveGatewayTenant();
  assert.equal(screen,'onboardScreen');assert.equal(c.sheetId,'');assert.equal(launched,false);
});
test('pending setup restores the saved operation name',()=>{
  const c=context();c.showGatewayOnboarding({name:'Saved Ranch',state:'INITIALIZING'});
  assert.equal(c.document.getElementById('operationInput').value,'Saved Ranch');
  assert.equal(c.document.getElementById('operationInput').disabled,true);
  assert.match(c.document.getElementById('onboardBtn').textContent,/Resume/);
});
test('setup sends only a name, suppresses double tap and re-resolves access',async()=>{
  let release,calls=0,resolved=0;
  const c=context({fetchWithAuthRetry:(url,options)=>{calls++;assert.equal(url,'https://example.invalid/onboarding');
    assert.deepEqual(JSON.parse(options.body),{name:'My Ranch'});
    return new Promise(r=>release=r);},resolveGatewayTenant:async()=>resolved++});
  const pending=c.createGatewayWorkspace();await c.createGatewayWorkspace();assert.equal(calls,1);
  release({ok:true,json:async()=>({status:'ACTIVE'})});await pending;
  assert.equal(resolved,1);assert.equal(c.onboardingBusy,false);
});
test('failed setup exposes retry and never launches the scanner',async()=>{
  let message,resolved=false;
  const c=context({showToast:m=>message=m,resolveGatewayTenant:async()=>resolved=true,
    fetchWithAuthRetry:async()=>({ok:false,json:async()=>({error:{message:'Retry later; original create is uncertain'}})})});
  await c.createGatewayWorkspace();assert.match(message,/uncertain/);assert.equal(resolved,false);
  assert.equal(c.document.getElementById('onboardBtn').disabled,false);
});
test('setup response after account switch cannot connect a Sheet',async()=>{
  let release,resolved=false;
  const c=context({fetchWithAuthRetry:()=>new Promise(r=>release=r),resolveGatewayTenant:async()=>resolved=true});
  const pending=c.createGatewayWorkspace();c.onboardingAccountEpoch++;
  release({ok:true,json:async()=>({status:'ACTIVE'})});await pending;assert.equal(resolved,false);
});
test('configuration response after account switch cannot launch an operation',async()=>{
  let release,launched=false;
  const c=context({launchApp:()=>launched=true,fetchWithAuthRetry:()=>new Promise(r=>release=r)});
  vm.runInContext(section('  function resolveGatewayTenant()', '  function launchApp()'),c);
  const pending=c.resolveGatewayTenant();c.onboardingAccountEpoch++;
  release({ok:true,json:async()=>({status:'ACTIVE',operation:{id:'a',name:'A',sheet_id:'a'},features:{field_head_count:true}})});
  await pending;assert.equal(launched,false);assert.equal(c.sheetId,'');
});
test('Google login uses a random state and S256 PKCE challenge',async()=>{
  const c=context();vm.runInContext(section('  function randomAuthValue()', '  function exchangeCodeForToken('),c);
  await c.window.signIn();const url=new URL(c.window.location.href);
  assert.equal(url.searchParams.get('state'),c.sessionStorage.getItem('hm_oauth_state'));
  assert.equal(url.searchParams.get('code_challenge_method'),'S256');
  assert.equal(url.searchParams.get('code_challenge'),require('node:crypto').createHash('sha256').update(c.sessionStorage.getItem('hm_oauth_verifier')).digest('base64url'));
});
test('OAuth callback with wrong state never exchanges the authorization code',()=>{
  let calls=0,message;
  const c=context({fetch:()=>{calls++;},showToast:m=>message=m});
  c.window.location.search='?code=test&state=wrong';
  c.sessionStorage.setItem('hm_oauth_state','expected');c.sessionStorage.setItem('hm_oauth_verifier','v'.repeat(64));
  vm.runInContext(section('  function exchangeCodeForToken(', '  var tokenRefreshPromise ='),c);
  c.exchangeCodeForToken('test');assert.equal(calls,0);assert.match(message,/could not be verified/);
});
test('saved email without a private device credential cannot silently refresh',async()=>{
  let calls=0;
  const c=context({fetch:()=>{calls++;}});
  vm.runInContext(section('  var tokenRefreshPromise =', '  function fetchWithAuthRetry('),c);
  await assert.rejects(c.refreshAccessToken('owner@example.com'),/Fresh Google sign-in/);
  assert.equal(calls,0);
});
