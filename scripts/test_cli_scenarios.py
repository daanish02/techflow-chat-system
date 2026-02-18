"""
CLI Test Script - 5 Conversation Scenarios for TechFlow Chat System

Tests:
1. Money Problems - Financial hardship, test retention offers
2. Phone Problems - Product defect, test retention vs cancellation
3. Questioning Value - Customer uncertain, test explanation and retention
4. Technical Help Needed - Technical issue, test tech support routing
5. Billing Question - Billing inquiry, test billing department routing
"""

import asyncio
from pathlib import Path
import sys
import time

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

TEST_SCENARIOS = [
    (
        "hey can't afford the $13/month care+ anymore, need to cancel",
        "sarah.chen@email.com",
        "retention",
        "Test 1: Money Problems - Should route to retention with discount/pause offers",
    ),
    (
        "this phone keeps overheating, want to return it and cancel everything",
        "mike.rodriguez@email.com",
        "retention",
        "Test 2: Phone Problems - Should route to retention and offer replacement",
    ),
    (
        "paying for care+ but never used it, maybe just get rid of it?",
        "lisa.kim@email.com",
        "retention",
        "Test 3: Questioning Value - Should stay in retention to explain value",
    ),
    (
        "my phone won't charge anymore, tried different cables",
        "james.wilson@email.com",
        "tech_support",
        "Test 4: Technical Help - Should route to tech support (no cancellation intent)",
    ),
    (
        "got charged $15.99 but thought care+ was $12.99, what's the extra?",
        "maria.garcia@email.com",
        "billing",
        "Test 5: Billing Question - Should route to billing department",
    ),
]


class CLITester:
    """Test the TechFlow Chat System CLI with predefined conversations."""

    def __init__(self, timeout: int = 45):
        """Initialize the CLI tester."""
        self.timeout = timeout
        self.process = None
        self.test_results = []

    async def start_cli(self) -> bool:
        """Start the CLI process."""
        try:
            self.process = await asyncio.create_subprocess_exec(
                "uv",
                "run",
                "cli_chat.py",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            print("[OK] CLI process started\n")
            return True
        except Exception as e:
            print(f"[FAIL] Failed to start CLI: {e}")
            return False

    async def wait_for_ready(self) -> bool:
        """Wait for CLI to be ready for input."""
        try:
            print("Waiting for CLI initialization...")
            await asyncio.sleep(5)
            print("✓ CLI ready for testing\n")
            return True
        except Exception as e:
            print(f"✗ Error waiting for CLI: {e}")
            return False

    async def send_input(self, message: str) -> str:
        """Send input to CLI and get response."""
        try:
            self.process.stdin.write((message + "\n").encode())
            await self.process.stdin.drain()

            output = b""
            start_time = time.time()

            while time.time() - start_time < self.timeout:
                try:
                    chunk = await asyncio.wait_for(
                        self.process.stdout.read(1024), timeout=2.0
                    )
                    if chunk:
                        output += chunk
                    else:
                        break
                except asyncio.TimeoutError:
                    break

            return output.decode("utf-8", errors="replace")

        except Exception as e:
            print(f"Error sending input: {e}")
            return ""

    async def run_test(
        self,
        customer_complaint: str,
        email: str,
        expected_routing: str,
        description: str,
    ) -> bool:
        """Run a single test scenario."""
        print(f"\n{'=' * 70}")
        print(description)
        print(f"{'=' * 70}")
        print(f"Customer: {customer_complaint}")
        print(f"Email: {email}\n")

        print("[Step 1] Sending initial greeting...\n")
        response = await self.send_input("hello")
        self._print_agent_response(response)

        print("\n[Step 2] Sending customer complaint...\n")
        response = await self.send_input(customer_complaint)
        self._print_agent_response(response)

        print("\n[Step 3] Providing email address...\n")
        response = await self.send_input(email)
        self._print_agent_response(response)

        routing_found = self._check_routing(response, expected_routing)

        if routing_found:
            print(f"\n[PASS] Routed to {expected_routing} as expected")
            self.test_results.append((description, True))
            return True
        else:
            print(f"\n[FAIL] Expected {expected_routing} routing not found")
            print(f"Raw response: {response[:200]}...")
            self.test_results.append((description, False))
            return False

    def _print_agent_response(self, response: str) -> None:
        """Print agent response, truncating if too long."""
        lines = response.split("\n")
        agent_lines = [l for l in lines if l.strip()]

        relevant = "\n".join(agent_lines[-5:]) if agent_lines else response
        print(relevant[:500])

    def _check_routing(self, response: str, expected_routing: str) -> bool:
        """Check if response matches expected routing."""
        response_lower = response.lower()

        if expected_routing == "retention":
            routing_keywords = [
                "retention",
                "offer",
                "discount",
                "pause",
                "upgrade",
                "solution",
                "help you keep",
            ]
            return any(keyword in response_lower for keyword in routing_keywords)

        elif expected_routing == "tech_support":
            routing_keywords = [
                "tech support",
                "technical support",
                "device team",
                "hardware",
                "transferring you to",
                "technical support team",
            ]
            return any(keyword in response_lower for keyword in routing_keywords)

        elif expected_routing == "billing":
            routing_keywords = [
                "billing",
                "billing department",
                "billing team",
                "charge",
                "payment",
                "invoice",
                "transferring you to billing",
            ]
            return any(keyword in response_lower for keyword in routing_keywords)

        return False

    async def stop_cli(self):
        """Stop the CLI process."""
        if self.process:
            try:
                self.process.stdin.write(b"exit\n")
                await self.process.stdin.drain()
                await asyncio.sleep(1)
            except Exception:
                pass

            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except (ProcessLookupError, asyncio.TimeoutError):
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass

    def print_summary(self):
        """Print test summary."""
        print(f"\n\n{'=' * 70}")
        print("TEST SUMMARY")
        print(f"{'=' * 70}\n")

        passed = sum(1 for _, result in self.test_results if result)
        total = len(self.test_results)

        for description, result in self.test_results:
            status = "PASS" if result else "FAIL"
            print(f"{status}: {description}")

        print(f"\n{'=' * 70}")
        print(f"Results: {passed}/{total} tests passed")
        print(f"{'=' * 70}\n")

        return passed == total

    async def run_all_tests(self) -> bool:
        """Run all test scenarios."""
        print("\n" + "=" * 70)
        print("TechFlow Chat System - CLI Test Suite")
        print("=" * 70)

        all_passed = True

        for customer_complaint, email, expected_routing, description in TEST_SCENARIOS:
            if not await self.start_cli():
                return False

            if not await self.wait_for_ready():
                return False

            test_passed = await self.run_test(
                customer_complaint, email, expected_routing, description
            )
            all_passed = all_passed and test_passed

            await self.stop_cli()
            await asyncio.sleep(1)

        self.print_summary()

        return all_passed


async def main():
    """Main entry point."""
    tester = CLITester(timeout=15)
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
