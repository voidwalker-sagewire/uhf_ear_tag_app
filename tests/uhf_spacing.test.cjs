const {test}=require('node:test');
const assert=require('node:assert/strict');
const {scanner}=require('./helpers/scanner.cjs');
test('DCC Ranch and Calf records match spaced UHF values without changing identifiers',async()=>{
  const c=scanner({fetchWithAuthRetry:async()=>({ok:true,json:async()=>({sheets:[]})})});
  c.loadTab=async()=>[{'Calf Tag':'917','UHF#':'740\t086\u00a0010 000 014'}];
  c.loadRanchTracker=async()=>[{source:'Ranch Tracker',tag:'133',uhf:'740 086 010 000 133'}];
  const calf={tag:'740086010000014'},ranch={tag:'740086010000133'};
  c.uniqueTags.calf=calf;c.uniqueTags.ranch=ranch;
  await c.lookupAnimal('calf',calf);await c.lookupAnimal('ranch',ranch);
  assert.equal(calf.animal.tag,'917');assert.equal(ranch.animal.tag,'133');
  assert.equal(ranch.animal.uhf,'740 086 010 000 133');
  assert.equal(ranch.tag,'740086010000133');
});
test('Animals workspace matches spaced UHF and keeps a genuinely unassigned tag unlinked',async()=>{
  const c=scanner({sheetId:'customer-sheet',fetchWithAuthRetry:async()=>({ok:true,json:async()=>({sheets:[{properties:{title:'Animals'}}]})})});
  c.loadTab=async()=>[{'Animal ID':'animal-1','Tag EPC':'740 086 010 000 133'}];
  const linked={tag:'740086010000133'},unlinked={tag:'740086010000017'};
  c.uniqueTags.linked=linked;c.uniqueTags.unlinked=unlinked;
  await c.lookupAnimal('linked',linked);await c.lookupAnimal('unlinked',unlinked);
  assert.equal(linked.animal.tag,'animal-1');assert.equal(unlinked.animal,null);
  assert.equal(unlinked.animalLookupError,null);
});
test('lookup preserves leading zeros, punctuation and exact printed-tag fallback',()=>{
  const c=scanner();
  const records=[{tag:'cow',uhf:'001 234'},{tag:'printed-17',uhf:''}];
  const good={tag:'001234'};c.applyAnimalMatch(records,good);assert.equal(good.animal.tag,'cow');
  for(const tag of ['1234','001-234','', '001235']){
    const r={tag};c.applyAnimalMatch(records,r);assert.equal(r.animal,null);
  }
  const printed={tag:'printed-17'};c.applyAnimalMatch(records,printed);assert.equal(printed.animal.tag,'printed-17');
});
