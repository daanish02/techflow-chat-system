"""Unit tests for LangChain data tools."""

from src.tools import (
    calculate_retention_offer,
    get_customer_data,
    update_customer_status,
)


class TestGetCustomerData:
    """Tests for get_customer_data tool."""

    def test_get_customer_valid_email(self):
        """Test retrieving customer with valid email."""
        result = get_customer_data.invoke({"email": "sarah.chen@email.com"})

        assert "error" not in result
        assert result["customer_id"] == "CUST_001"
        assert result["name"] == "Sarah Chen"
        assert result["tier"] == "premium"
        assert result["monthly_charge"] == 12.99

    def test_get_customer_case_insensitive(self):
        """Test email lookup is case-insensitive."""
        result = get_customer_data.invoke({"email": "SARAH.CHEN@EMAIL.COM"})

        assert "error" not in result
        assert result["customer_id"] == "CUST_001"

    def test_get_customer_invalid_email(self):
        """Test handling of non-existent customer."""
        result = get_customer_data.invoke({"email": "nonexistent@email.com"})

        assert result["error"] == "Customer not found"
        assert "message" in result

    def test_get_customer_multiple_tiers(self):
        """Test retrieving customers from different tiers."""
        tiers_to_test = [
            ("mike.rodriguez@email.com", "new"),
            ("james.wilson@email.com", "regular"),
            ("sarah.chen@email.com", "premium"),
            ("michael.davis@email.com", "premium"),
        ]

        for email, expected_tier in tiers_to_test:
            result = get_customer_data.invoke({"email": email})
            assert result["tier"] == expected_tier


class TestCalculateRetentionOffer:
    """Tests for calculate_retention_offer tool."""

    def test_calculate_offer_financial_hardship(self):
        """Test offer calculation for financial hardship."""
        result = calculate_retention_offer.invoke(
            {"customer_tier": "premium", "reason": "financial_hardship"}
        )

        assert "error" not in result
        assert result["tier"] == "premium"
        assert result["reason"] == "financial_hardship"
        assert len(result["offers"]) == 2
        assert result["strategy"]["primary"] == "pause"
        assert result["strategy"]["secondary"] == "discount"

    def test_calculate_offer_product_defect(self):
        """Test offer calculation for product defect."""
        result = calculate_retention_offer.invoke(
            {"customer_tier": "regular", "reason": "product_defect"}
        )

        assert "error" not in result
        assert result["tier"] == "regular"
        assert len(result["offers"]) > 0

    def test_calculate_offer_not_using(self):
        """Test offer calculation when customer isn't using service."""
        result = calculate_retention_offer.invoke(
            {"customer_tier": "new", "reason": "not_using"}
        )

        assert "error" not in result
        assert result["tier"] == "new"
        assert len(result["offers"]) > 0

    def test_tier_discount_percentages(self):
        """Test that different tiers get different offers."""
        tiers_and_offers = [
            ("new", 1),  # new customers get 1 offer
            ("regular", 2),  # regular customers get 2 offers
            ("premium", 2),  # premium customers get 2 offers
        ]

        for tier, expected_count in tiers_and_offers:
            result = calculate_retention_offer.invoke(
                {"customer_tier": tier, "reason": "financial_hardship"}
            )

            assert len(result["offers"]) == expected_count
            assert "error" not in result

    def test_invalid_tier_defaults_to_new(self):
        """Test that invalid tier defaults to new."""
        result = calculate_retention_offer.invoke(
            {"customer_tier": "invalid_tier", "reason": "other"}
        )

        assert result["tier"] == "invalid_tier"
        assert "error" not in result

    def test_unknown_reason_defaults_to_other(self):
        """Test that unknown reason is handled."""
        result = calculate_retention_offer.invoke(
            {"customer_tier": "premium", "reason": "some_random_reason"}
        )

        assert result["reason"] == "some_random_reason"
        assert "error" not in result
        assert len(result["offers"]) > 0


class TestUpdateCustomerStatus:
    """Tests for update_customer_status tool."""

    def test_update_status_cancelled(self):
        """Test logging cancellation."""
        result = update_customer_status.invoke(
            {
                "customer_id": "CUST_001",
                "action": "cancelled_insurance",
                "details": "reason: financial_hardship",
            }
        )

        assert result["success"] is True
        assert result["customer_id"] == "CUST_001"
        assert result["action"] == "cancelled_insurance"
        assert "timestamp" in result

    def test_update_status_accepted_discount(self):
        """Test logging discount acceptance."""
        result = update_customer_status.invoke(
            {
                "customer_id": "CUST_002",
                "action": "accepted_discount",
                "details": "50% off for 6 months",
            }
        )

        assert result["success"] is True
        assert "50% off for 6 months" in result["log_entry"]

    def test_update_status_without_details(self):
        """Test logging action without details."""
        result = update_customer_status.invoke(
            {"customer_id": "CUST_003", "action": "kept_coverage", "details": ""}
        )

        assert result["success"] is True
        assert result["customer_id"] == "CUST_003"

    def test_log_file_created(self):
        """Test that log file is created."""
        from src.config import settings

        # Perform an update
        update_customer_status.invoke(
            {
                "customer_id": "CUST_999",
                "action": "test_action",
                "details": "test details",
            }
        )

        log_file = settings.DATA_DIR / "logs" / "customer_updates.log"
        assert log_file.exists()

        # Verify content
        content = log_file.read_text()
        assert "CUST_999" in content
        assert "test_action" in content


class TestToolsIntegration:
    """Integration tests for tool workflows."""

    def test_complete_retention_workflow(self):
        """Test complete workflow: get customer -> calculate offer -> update status."""
        # get customer
        customer = get_customer_data.invoke({"email": "sarah.chen@email.com"})
        assert "error" not in customer

        # calculate offer
        offers = calculate_retention_offer.invoke(
            {"customer_tier": customer["tier"], "reason": "financial_hardship"}
        )
        assert len(offers["offers"]) > 0

        # update status
        result = update_customer_status.invoke(
            {
                "customer_id": customer["customer_id"],
                "action": "accepted_discount",
                "details": offers["offers"][0]["description"],
            }
        )
        assert result["success"] is True

    def test_cancellation_workflow(self):
        """Test cancellation workflow."""
        # get customer
        customer = get_customer_data.invoke({"email": "james.wilson@email.com"})

        # calculate offers (agent tried to retain)
        calculate_retention_offer.invoke(
            {"customer_tier": customer["tier"], "reason": "too_expensive"}
        )

        # customer declines, process cancellation
        result = update_customer_status.invoke(
            {
                "customer_id": customer["customer_id"],
                "action": "cancelled_insurance",
                "details": "customer declined all offers",
            }
        )
        assert result["success"] is True
