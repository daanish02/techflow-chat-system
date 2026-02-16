"""LangChain tools for customer data access."""

from typing import Any, Dict

import pandas as pd
from langchain_core.tools import tool

from src.config import settings
from src.logger import get_logger

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
