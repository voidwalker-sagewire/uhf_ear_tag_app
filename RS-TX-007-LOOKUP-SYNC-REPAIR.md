# RS-TX-007: lookup recovery and durable scan upload repair

Status: owner-authorized repair candidate, October 22, 2026. Review/merge,
deployment, live Google integration and C72 acceptance remain pending.

## Problem and resulting behavior

A failed lookup used to remain failed when the same tag was scanned again.
A later successful load could show another animal while the earlier card stayed
in error. Repeated reads increased local totals without updating the saved row.
Independent live writes and overlapping offline flushes could append the same
observation and then remove another queued item without sending it.

The repair uses one durable queue for new scans and count updates, one active
drain per page, and a Web Lock for drains across supporting tabs. Observations
receive a stable HCID, original session, destination, operation and author before
upload. Queue acknowledgements target the ID rather than queue position. Reads
arriving during a request remain queued. Count changes coalesce and update column
J of the existing HeadCounter row.

Uncertain appends are recorded before the request. Retry reads the HCID column
before deciding whether to append or update. A cached row location is checked
against its HCID before updating it. Unknown-owner legacy entries stay held.
Eligible older entries retain unknown session/author as blank rather than
attributing them to a later sign-in. Existing duplicates are not deleted.

Animal reads have one bounded retry for transient request failures, stage/status
diagnostics, up to three automatic lookup attempts, same-tag retry after a short
backoff, and recovery on reconnection. A successful full load repairs earlier
failed cards. DCC still uses Calf Tracker plus Ranch Tracker; a failed source
cannot be interpreted as an unlinked animal. New workspaces still use Animals.
A network failure after a successful Google token refresh remains a network error.

## Verification and limits

Run:

- node --test tests/*.test.cjs
- python -m unittest discover -s tests
- python tools/build_gateway_page.py
- git diff --check

Runtime regressions use the actual entry-point functions with synthetic Google
responses, storage and timers. They cover lost append replies/restart recovery,
in-flight count changes, row movement, duplicate IDs, rejected reconciliation,
account changes, overlapping drains, Beacon target preservation, storage failure,
bounded lookup recovery and the legacy 917/933 mapping shape.

This does not provide a server-side exactly-once Google Sheets transaction.
Manual row deletion/sorting during a request, late remote commits after an
uncertain response, and concurrent edits of shared localStorage remain limits.
Keep one active scanner tab for capture; Web Locks coordinate drains, not atomic
multi-tab capture/storage writes. Browsers without Web Locks retain page-local
serialization. Older queued rows lacking the original HCID cannot be
retroactively deduplicated. Existing historic counts are not rewritten.

The original deleted device error cannot be reconstructed from a unit test.
No live credentials or production Sheet writes are used by these regressions.
No C72, real offline/reconnect, browser visual QA or deployed-source verification
is claimed by this candidate.

## Deployment and owner test

After review/merge, deploy the scanner resource serving scanner.herdmate.ag;
verify the served gateway.html contains the repair before asking for field work.
Preserve site storage and unsynced queues. No new APK/AAB or Play release is part
of this repair.

First owner check: confirm personal sign-in and DCC, then scan 0014 once.
Expected animal: 917. If a failure occurs, immediately copy the diagnostic log.
Only after that result, conduct the controlled reconnect/count-save checks and
read back the DCC destination. Do not send the owner on another general scan.

Authoritative engineering ledger:
https://drive.google.com/file/d/1XDmvwWZwLfcMwVmmmPKQf3TrZziCr7lA/view

Central register:
https://docs.google.com/spreadsheets/d/1AXDlNFjZKbh5sAV_J7ujp3yKzRyJ6AHaNDkhDiNsQ_o/edit

