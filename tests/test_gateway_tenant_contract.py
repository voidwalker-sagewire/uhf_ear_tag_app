from pathlib import Path
import unittest


INDEX = (Path(__file__).parents[1] / "index.html").read_text(encoding="utf-8")


class GatewayTenantContractTest(unittest.TestCase):
    def test_gateway_route_is_explicit_and_fail_closed(self):
        self.assertIn("tenant') === 'gateway'", INDEX)
        self.assertIn("SCOUT_CONFIG_API", INDEX)
        self.assertIn("data.status!=='ACTIVE'", INDEX)
        self.assertIn("ONBOARDING_REQUIRED", INDEX)

    def test_gateway_sheet_comes_from_verified_operation(self):
        self.assertIn("saveLocal(String(data.operation.sheet_id)", INDEX)
        self.assertIn("operationId=String(data.operation.id)", INDEX)
        self.assertIn("operationFeatures.field_head_count!==true", INDEX)

    def test_sheet_requests_use_resolved_sheet(self):
        request_lines = [line for line in INDEX.splitlines() if "sheets.googleapis.com" in line]
        self.assertTrue(request_lines)
        self.assertFalse(any("DCC_SHEET_ID" in line for line in request_lines))

    def test_offline_queue_is_operation_scoped(self):
        self.assertIn("operationId:operationId||null", INDEX)
        self.assertIn("q[0].operationId !== operationId", INDEX)
        self.assertIn("Legacy offline queue held", INDEX)

    def test_legacy_route_remains_available_for_code12(self):
        self.assertIn("saveLocal(DCC_SHEET_ID, 'DCC')", INDEX)


if __name__ == "__main__":
    unittest.main()
