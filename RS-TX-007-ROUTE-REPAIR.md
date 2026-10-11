# RS-TX-007 — dedicated Gateway route repair

The live custom-domain root fetched on 2026-10-11 did not contain Gateway routing, although repository main did. The Oct 8 clean-state DCC result therefore does not prove the Gateway contract ran.

`gateway.html?mode=headcount` is a complete scanner page with a visible source deployment marker. It chooses Gateway mode from its path even after site data is cleared. Google OAuth retains the already registered root redirect URI; the persisted Gateway marker carries that route through the callback. Sign-out preserves that route. Tenant resolution clears stale in-memory tenant information before requesting configuration.

The page is generated from `index.html` by `python tools/build_gateway_page.py`; the runtime suite rejects a stale generated copy. Run `node --test tests/gateway_runtime.test.cjs` and `python -m unittest discover -s tests` before publication.

Deployment acceptance requires fetching `https://scanner.herdmate.ag/gateway.html` and finding `gateway-rs-tx-007` before distributing a wrapper that opens that exact route. The existing custom domain's publishing source needs verification because its root differs from repository main. The formerly recorded Android repository currently returns 404; the older `tag_reaper-scout` project is not the code13 wrapper and must not be substituted.

Device acceptance: clear site data, open the dedicated page, sign in to the authorized operation, and require both `Gateway tenant resolution started` and `Gateway tenant resolved` in the debug log. Repeat using a separate operation/account and confirm a distinct authorized Sheet. Unknown/disabled accounts must remain blocked. Queued scans belonging to the first operation must not upload to the second. These live checks remain pending; mocked tests do not constitute field acceptance.
