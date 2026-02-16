"""LangChain tools for customer data access."""

import json
from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from src.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@tool
def get_customer_data(email: str) -> Dict[str, Any]:
    """
    Retrieve customer profile information by email address.

    This tool loads customer data from the CSV database and returns
    the customer's profile including tier, tenure, and current plan details.
    Use this tool to authenticate customers and gather information needed
    for retention offers.

    Args:
        email: Customer's email address

    Returns:
        Dictionary containing customer data:
        - customer_id: Unique identifier
        - name: Full name
        - email: Email address
        - phone: Phone number
        - plan_type: Insurance plan
        - monthly_charge: Current monthly insurance cost
        - signup_date: Date customer signed up
        - status: Account status
        - total_spent: Total amount spent to date
        - support_tickets_count: Number of support tickets filed
        - account_health_score: Health score
        - tenure_months: Months as customer
        - tier: Loyalty tier (new, regular, premium)
        - device: Current device model
        - purchase_date: Date of device purchase

    Example:
        >>> get_customer_data("sarah.chen@email.com")
        {
            "customer_id": "CUST_001",
            "name": "Sarah Chen",
            "email": "sarah.chen@email.com",
            "phone": "555-0101",
            "plan_type": "Care+ Premium",
            "monthly_charge": 12.99,
            "signup_date": "2023-06-15",
            "status": "Active",
            "total_spent": 1299.00,
            "support_tickets_count": 2,
            "account_health_score": 85,
            "tenure_months": 8,
            "tier": "premium",
            "device": "Pro Max",
            "purchase_date": "2023-06-15"
        }
    """
    try:
        # load customer database
        csv_path = settings.DATA_DIR / "customers.csv"

        if not csv_path.exists():
            logger.error(f"Customer database not found at {csv_path}")
            return {"error": "Customer database not available", "email": email}

        df = pd.read_csv(csv_path)

        # search for customer by email
        customer = df[df["email"].str.lower() == email.lower()]

        if customer.empty:
            logger.warning(f"No customer found with email: {email}")
            return {
                "error": "Customer not found",
                "email": email,
                "message": "Please verify the email address and try again.",
            }

        # convert to dictionary
        customer_data = customer.iloc[0].to_dict()

        logger.info(
            f"Retrieved customer data for {customer_data['name']} "
            f"(tier: {customer_data['tier']})"
        )

        return customer_data

    except Exception as e:
        logger.error(f"Error retrieving customer data: {e}")
        return {
            "error": "System error",
            "email": email,
            "message": "Unable to retrieve customer information at this time.",
        }


@tool
def calculate_retention_offer(customer_tier: str, reason: str) -> Dict[str, Any]:
    """
    Calculate personalized retention offers based on customer tier and cancellation reason.

    This tool uses business rules to generate appropriate retention offers.
    It considers the customer's loyalty tier and their stated reason for cancellation to provide the most relevant incentives.

    Args:
        customer_tier: Customer's loyalty tier (premium, regular, new)
        reason: Cancellation reason - one of:
            - "financial_hardship": Cannot afford current cost
            - "product_defect": Issues with the phone/device (overheating, battery, etc.)
            - "not_using": Haven't used the insurance benefits
            - "too_expensive": Price too high in general
            - "switching_carrier": Moving to different carrier
            - "other": Other reasons

    Returns:
        Dictionary containing:
        - offers: List of personalized offers
        - tier: Customer tier used for calculation
        - reason: Reason category matched
        - strategy: Summary of the offer strategy

    Example:
        >>> calculate_retention_offer("premium", "financial_hardship")
        {
            "offers": [
                {
                    "type": "pause",
                    "description": "Pause subscription for 6 months with no charges",
                    "authorization": "agent"
                },
                {
                    "type": "discount",
                    "description": "50% discount for full year",
                    "authorization": "manager"
                }
            ],
            "tier": "premium",
            "reason": "financial_hardship",
            "strategy": {
                "primary": "pause",
                "secondary": "discount"
            }
        }
    """
    try:
        # load retention rules
        rules_path = settings.DATA_DIR / "retention_rules.json"

        if not rules_path.exists():
            logger.error(f"Retention rules not found at {rules_path}")
            return {
                "error": "Retention rules not available",
                "tier": customer_tier,
                "reason": reason,
            }

        with open(rules_path, "r") as f:
            rules = json.load(f)

        # normalize inputs
        tier = customer_tier.lower().strip()
        reason_key = reason.lower().strip().replace(" ", "_")

        # map reason to a JSON category
        reason_to_category = {
            "financial_hardship": "financial_hardship",
            "too_expensive": "financial_hardship",
            "product_defect": "product_issues",
            "not_using": "service_value",
            "switching_carrier": "service_value",
            "other": "service_value",
        }

        category = reason_to_category.get(reason_key, "service_value")

        if category not in rules:
            logger.warning(
                f"Category '{category}' not in rules, falling back to 'service_value'"
            )
            category = "service_value"

        category_rules = rules[category]

        # map tier to the JSON customer segment
        tier_to_segment = {
            "premium": "premium_customers",
            "regular": "regular_customers",
            "new": "new_customers",
        }

        segment = tier_to_segment.get(tier, "new_customers")

        # resolve offers from the category
        offers = _resolve_offers(category_rules, segment, reason_key)

        if not offers:
            logger.warning(
                f"No offers found for category={category}, segment={segment}. "
                f"Using fallback."
            )
            fallback_rules = rules.get("financial_hardship", {})
            offers = fallback_rules.get("new_customers", [])

        logger.info(
            f"Generated {len(offers)} retention offers for tier={tier} "
            f"(segment={segment}), reason={reason_key} (category={category})"
        )

        # build strategy summary from the first two offers
        strategy = {}
        if len(offers) >= 1:
            strategy["primary"] = offers[0].get("type", "unknown")
        if len(offers) >= 2:
            strategy["secondary"] = offers[1].get("type", "unknown")

        return {
            "offers": offers,
            "tier": tier,
            "reason": reason_key,
            "strategy": strategy,
        }

    except Exception as e:
        logger.error(f"Error calculating retention offer: {e}")
        return {
            "error": "System error",
            "tier": customer_tier,
            "reason": reason,
            "message": "Unable to calculate retention offers at this time.",
        }


def _resolve_offers(
    category_rules: Dict[str, Any], segment: str, reason_key: str
) -> list:
    """
    Resolve the list of offers from a category's rules.

    Tries segment-based lookup first (e.g. premium_customers),
    then reason-key matching, and finally returns the
    first available list of offers as a fallback.

    Args:
        category_rules: The dict under the matched top-level category
        segment: The customer segment key (e.g. "premium_customers")
        reason_key: The original normalized reason string

    Returns:
        List of offer dicts
    """
    # direct segment match
    if segment in category_rules:
        value = category_rules[segment]
        if isinstance(value, list):
            return value

    # try matching reason
    if reason_key in category_rules:
        value = category_rules[reason_key]
        if isinstance(value, list):
            return value

    # 3) return the first list we find in the category
    for key, value in category_rules.items():
        if isinstance(value, list):
            return value

    return []


@tool
def update_customer_status(
    customer_id: str, action: str, details: str = ""
) -> Dict[str, Any]:
    """
    Process and log customer status updates.

    This tool records all customer decisions and actions to an audit log.
    Use this tool when a customer makes a final decision about their coverage,
    whether they accept an offer, cancel their service, or pause their coverage.

    Args:
        customer_id: Customer's unique identifier
        action: Action taken - one of:
            - "cancelled_insurance": Customer cancelled coverage
            - "accepted_discount": Customer accepted discount offer
            - "accepted_pause": Customer accepted pause offer
            - "accepted_upgrade": Customer accepted upgrade offer
            - "kept_coverage": Customer decided to keep coverage as-is
            - "escalated": Issue escalated to specialist
        details: Additional details about the action (reason, offer terms, etc.)

    Returns:
        Dictionary containing:
        - success: Whether the update was logged successfully
        - customer_id: Customer identifier
        - action: Action logged
        - timestamp: When the action was recorded
        - message: Confirmation message

    Example:
        >>> update_customer_status("C001", "accepted_discount", "50% off for 6 months")
        {
            "success": True,
            "customer_id": "C001",
            "action": "accepted_discount",
            "timestamp": "2026-02-16 14:23:45",
            "message": "Customer status updated successfully"
        }
    """
    try:
        from datetime import datetime

        # create logs directory if it doesn't exist
        logs_dir = settings.DATA_DIR / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        log_file = logs_dir / "customer_updates.log"

        # generate timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # format log entry
        log_entry = f"{timestamp} | {customer_id} | {action}"
        if details:
            log_entry += f" | {details}"
        log_entry += "\n"

        # append to log file
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_entry)

        logger.info(f"Logged action for {customer_id}: {action}")

        return {
            "success": True,
            "customer_id": customer_id,
            "action": action,
            "timestamp": timestamp,
            "message": "Customer status updated successfully",
            "log_entry": log_entry.strip(),
        }

    except Exception as e:
        logger.error(f"Error updating customer status: {e}")
        return {
            "success": False,
            "customer_id": customer_id,
            "action": action,
            "error": str(e),
            "message": "Failed to update customer status. Please try again.",
        }
