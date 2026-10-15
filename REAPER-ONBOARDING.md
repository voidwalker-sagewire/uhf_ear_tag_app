# Shared Reaper onboarding client

Candidate authorized October 15, 2026; not yet deployed or device accepted.
All scanner entry points, including a fresh root launch from an Android wrapper,
now resolve the authenticated identity through the Gateway. DCC retains its
existing workbook and legacy animal loader; there is no direct-login DCC fallback.

Unknown accounts enter operation setup. POST /reaper/api/v1/onboarding sends
only an operation name. The Gateway creates/resumes the customer-owned workbook,
verifies its schema/ownership and persists membership. Scout then calls shared
/reaper/api/v1/config again; only an ACTIVE verified operation can scan. A pending
operation restores its original name. Duplicate taps are suppressed, failures
remain retryable, and responses arriving after sign-out/account change are held.
Existing operation-scoped offline queues remain held across account changes.

Authentication now uses OAuth state and S256 PKCE. A private device-session
credential from the auth service is required for silent refresh; email alone is
insufficient. Existing devices need one fresh sign-in after the coordinated
update. The unused Google Identity Services token-client path was removed so
it cannot override or bypass the verified code/session flow.

Deployment requires BOTH backend changes:
- voidwalker-sagewire/reaper_scout_auth, codex/reaper-shared-onboarding
- voidwalker-sagewire/sagewire-api-gateway, codex/reaper-shared-onboarding

The Gateway's REAPER-ONBOARDING.md contains the deployment/storage runbook. Keep
its existing static registry, configure a persistent local /data volume and
SAGEWIRE_REAPER_DB_PATH, preserve/back up the auth database, and coordinate auth
and scanner publication. Deploying only this frontend will not enable setup.
Do not restore insecure email-only refresh to accommodate an old cached client.

Validation: node --test tests/*.test.cjs; python -m unittest discover -s tests.
CI now runs every Node test file, including the existing Animals workspace tests.
Browser tests use simulated responses and do not replace real Google/C72 tests.
Test a separate new Google identity, then DCC/demo account isolation, restart,
Sheet ownership, known-tag lookup and scan-write destination before acceptance.
In-app animal creation, invitations/employee roles, reporting templates and R6
hardware decoding are not claimed complete by this account-provisioning change.
