const fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const source=fs.readFileSync(path.join(__dirname,'..','..','index.html'),'utf8');
function section(a,b){const p=source.indexOf(a),q=source.indexOf(b,p+1);if(p<0||q<p)throw Error('Missing source section '+a);return source.slice(p,q);}
function scanner(extra={}){
  const storage=new Map(),timers=[],events={};
  const c=vm.createContext({Promise,String,Date,encodeURIComponent,console,
    window:{addEventListener:(name,fn)=>events[name]=fn},navigator:{},localStorage:{getItem:k=>storage.get(k)||null,setItem:(k,v)=>storage.set(k,v)},
    document:{getElementById:()=>({className:''})},setTimeout:(fn,ms)=>{timers.push({fn,ms});return timers.length;},clearTimeout(){},
    DCC_SHEET_ID:'dcc-sheet',sheetId:'dcc-sheet',operationId:'dcc',operationName:'DCC',signedInEmail:'owner@example.com',sessionId:'session-original',
    accessToken:'fake',onboardingAccountEpoch:0,gatewayTenantMode:true,operationFeatures:{animal_lookup:true},
    bbTarget:'',uniqueTags:{},currentPasture:'Barn',lastGPS:null,
    newHCID:()=> 'H-'+require('node:crypto').randomUUID(),newBeaconID:()=> 'B-'+require('node:crypto').randomUUID(),
    debugLog(){},setSheetsBar(){},renderTable(){},renderTabs(){},updateStats(){},updateHeadcountBanner(){},showToast(){},playScanTick(){},markAuthExpired(){},
    ...extra});
  vm.runInContext(section('  function headCountRow(', '  var sheetId ='),c);
  vm.runInContext(section('  var QUEUE_KEY =','  // ── CHROME CHECK'),c);
  vm.runInContext(section('  function writeToSheet(', '  // updateSheetWeather removed:'),c);
  vm.runInContext(section('  function processHCTag(', '  // Animals is the new-account schema.'),c);
  vm.runInContext(section('  var animalRecordsCache =','  window.showAnimalModal ='),c);
  c.storage=storage;c.timers=timers;c.events=events;
  return c;
}
module.exports={scanner,section,source,tick:()=>new Promise(setImmediate)};

