const {test}=require('node:test');
const assert=require('node:assert/strict');
const vm=require('node:vm');
const {scanner,section,tick}=require('./helpers/scanner.cjs');
function setup(extra={}){
  const els={};
  const c=scanner({currentMode:'hc',bleBuffer:'',lastScanTime:{},accessToken:null,
    document:{getElementById:id=>els[id]||(els[id]={value:'',textContent:'',focus(){}})},...extra});
  vm.runInContext(section('  // ── PROCESS INCOMING TAG','  // SETTINGS'),c);
  return {c,els};
}
test('paused wedge ignores both valid and letter-prefixed input before logging or debounce',()=>{
  let logged=0;const {c,els}=setup({accessToken:'token',debugLog(){logged++;}});
  c.window.toggleHeadCountCapture();logged=0;
  els.tagInput.value='EA1234';let prevented=0;
  c.handleHeadCountKeydown({key:'E',preventDefault(){prevented++;}});
  c.processIncomingTag('EA12345678');c.processIncomingTag('740086010000258');
  assert.equal(els.tagInput.value,'');assert.equal(els.tagInput.readOnly,true);
  assert.equal(prevented,1);assert.equal(logged,0);assert.equal(Object.keys(c.uniqueTags).length,0);
  c.window.toggleHeadCountCapture();c.processIncomingTag('740086010000258');
  assert.equal(els.tagInput.readOnly,false);assert.equal(Object.keys(c.uniqueTags).length,1);
});
test('pause and resume clear BLE fragments; paused BLE is discarded before decoding',()=>{
  const {c}=setup();vm.runInContext(section('  function onBleData(', '  function setBleBar('),c);
  c.bleBuffer='74008601';c.window.toggleHeadCountCapture();assert.equal(c.bleBuffer,'');
  c.onBleData({});assert.equal(c.bleBuffer,''); // no decoder or event payload needed when paused
  c.bleBuffer='EA';c.window.toggleHeadCountCapture();assert.equal(c.bleBuffer,'');
});
test('progress distinguishes upload observations, pending repeats and other destinations',()=>{
  const {c,els}=setup();c.processHCTag('740086010000258');c.processHCTag('740086010000258');
  c.updateCaptureProgress();assert.match(els.captureProgress.textContent,/1 saved observation waiting for upload/);
  assert.match(els.captureProgress.textContent,/Repeat counts for 1 tags waiting/);
  c.queueAdd({tag:'other',readCount:1},'Head Count','other');c.window.toggleHeadCountCapture();c.updateCaptureProgress();
  assert.match(els.captureProgress.textContent,/PAUSED/);assert.match(els.captureProgress.textContent,/another destination/);
  assert.doesNotMatch(els.captureProgress.textContent,/Repeat counts/);
  assert.match(els.captureProgress.textContent,/Reader backlog is not available/);
});
test('unreadable queue cannot be reported as empty or fully synced',()=>{
  const {c,els}=setup();c.storage.set('hm_offline_queue','invalid');c.updateCaptureProgress();
  assert.match(els.captureProgress.textContent,/unreadable/);assert.doesNotMatch(els.captureProgress.textContent,/queue is empty/);
});
test('pausing does not cancel a saved observation upload already in flight',async()=>{
  let release;const {c,els}=setup({accessToken:'token',fetchWithAuthRetry:()=>new Promise(r=>release=r)});
  c.processHCTag('740086010000258');await tick();const pending=c.queueFlushPromise;
  c.window.toggleHeadCountCapture();c.updateCaptureProgress();assert.match(els.captureProgress.textContent,/sync working/);
  release({ok:true,json:async()=>({updates:{updatedRange:'HeadCounter!A2:M2'}})});await pending;
  assert.equal(c.queueLoad().length,0);assert.equal(c.headCountPaused,true);
  c.updateCaptureProgress();assert.match(els.captureProgress.textContent,/Saved upload queue is empty/);
});
