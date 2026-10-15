# New-account Animals workspace

Owner direction, 2026-10-12: use a single species-neutral Animals table for
new accounts. Existing DCC/HerdMate split trackers remain legacy sources.

## Contract

New workbook provisioning uses `workspaceDefinition()` and `workspaceHeaders()`
in index.html: Animals (A:P) and HeadCounter (A:M). HeadCounter headers match
the existing scan writer. Do not create Ranch Tracker, Calf Tracker or a
Bovine Beacon table for new workspaces. Treatments is deferred, not provisioned.

Animals headers: Tag EPC, Animal ID, Printed Tag, Name, Species, Sex, Breed,
DOB, Color, Description, Dam ID, Sire ID, Location, Status, Weight, Notes.
Identifiers are text. Parent references use Animal ID. Scan history remains
an event log, not duplicate animal identity rows. Additional tables are undecided.

Lookup selects Animals when present, even when empty. Existing workbooks with
both split trackers retain their loader; DCC's established Sheet explicitly
retains its legacy loader. Missing tables, permission failures or network errors
must not trigger a fallback into another operation. Old eight-column Animals
workbooks are readable through header aliases. Concurrent lookups share one
fetch, and failed fetches can be retried.

## Publication and onboarding limits

welcome.html is the public front door. index.html redirects only the root of
herdmate.ag, without an OAuth code, to it. scanner.herdmate.ag and its OAuth
callback are unchanged. help.html documents current behavior and gaps.
The live root page matched this repository's index.html byte for byte before
the edit. Confirm the actual Coolify resource serving herdmate.ag before
deploying; a source match alone does not prove its deployment wiring.

The generic sagewire-onboarding service does not provision these Sheets.
The shared Reaper Gateway onboarding candidate now handles automatic customer
provisioning and verified registration. Scout enters setup on ONBOARDING_REQUIRED
and re-resolves Gateway access before scanning. Publication and a fresh-customer
Google/C72 integration test remain pending. An in-app animal-creation form is
still a separate unfinished workflow. See REAPER-ONBOARDING.md.

Publish the repository changes, merge, redeploy the appropriate Coolify
resources, then verify the root welcome page, the guide, DCC lookup and a
separate Animals workspace with known tags. No production Sheets were edited.
