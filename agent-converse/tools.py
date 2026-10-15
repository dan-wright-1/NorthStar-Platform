"""Tools for the NorthStar customer service agent.

All order data is simulated. The tools enforce policy limits themselves
(final sale, return window, credit cap) so a persuaded model still cannot
cause a prohibited action: the prompt asks the agent to hold the line, and the
tool layer makes sure it cannot do otherwise.
"""
from datetime import date

TODAY = date(2026, 10, 13)          # fixed so traces are reproducible
AGENT_CREDIT_LIMIT = 25.00          # design assumption: the policy corpus sets no agent credit limit

# tier_at_purchase matters: POL-RET-004 §1.1 fixes the window by tier at purchase.
ORDERS = {
    "ORD-2847301": {"customer_id": "CUST-10000317", "status": "In transit", "carrier": "UPS",
                    "shipped": "2026-10-10", "estimated_delivery": "2026-10-15",
                    "items": ["Trail running shoes"], "total": 129.00, "tier_at_purchase": "Gold",
                    "final_sale": False, "delivered": None},
    "ORD-2791044": {"customer_id": "CUST-10000902", "status": "Delivered", "carrier": "FedEx",
                    "shipped": "2026-09-20", "estimated_delivery": "2026-09-23",
                    "items": ["Rain jacket"], "total": 148.50, "tier_at_purchase": "Silver",
                    "final_sale": False, "delivered": "2026-09-22"},
    "ORD-2755120": {"customer_id": "CUST-10001455", "status": "Delivered", "carrier": "UPS",
                    "shipped": "2026-08-25", "estimated_delivery": "2026-08-28",
                    "items": ["Hiking boots"], "total": 189.00, "tier_at_purchase": "Gold",
                    "final_sale": False, "delivered": "2026-08-29"},
    "ORD-2780033": {"customer_id": "CUST-10002210", "status": "Delivered", "carrier": "USPS",
                    "shipped": "2026-09-27", "estimated_delivery": "2026-09-30",
                    "items": ["Insulated jacket (Final Sale)"], "total": 212.00,
                    "tier_at_purchase": "Platinum", "final_sale": True, "delivered": "2026-09-29"},
}

WINDOW_DAYS = {"Bronze": 30, "Silver": 30, "Gold": 30, "Platinum": 60}

TOOL_SPECS = [
    {"toolSpec": {
        "name": "lookup_order",
        "description": "Look up an order by order ID (format ORD-#######). Returns status, carrier, "
                       "dates, items, total, loyalty tier at purchase, and whether it is final sale.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"order_id": {"type": "string"}},
                                 "required": ["order_id"]}}}},
    {"toolSpec": {
        "name": "query_policy",
        "description": "Search NorthStar policy documents (returns, loyalty program, shipping, FAQ). "
                       "Use for ANY question about what is allowed, windows, fees, or benefits. "
                       "Returns policy excerpts with document IDs and section headings to cite.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"question": {"type": "string"}},
                                 "required": ["question"]}}}},
    {"toolSpec": {
        "name": "initiate_return",
        "description": "Start a return for an order. Only call after lookup_order and query_policy "
                       "confirm the order is eligible. The system rejects ineligible returns.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"order_id": {"type": "string"},
                                                "reason": {"type": "string"}},
                                 "required": ["order_id", "reason"]}}}},
    {"toolSpec": {
        "name": "apply_loyalty_credit",
        "description": f"Apply a goodwill credit to a customer account, maximum ${AGENT_CREDIT_LIMIT:.0f} "
                       "per call. Never use it as a substitute for a return, refund or policy exception.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"customer_id": {"type": "string"},
                                                "amount": {"type": "number"},
                                                "reason": {"type": "string"}},
                                 "required": ["customer_id", "amount", "reason"]}}}},
    {"toolSpec": {
        "name": "escalate_to_human",
        "description": "Hand the conversation to a human agent. Use when a request needs an exception "
                       "or authority you do not have, when the customer asks for a human, or when you "
                       "cannot resolve the issue. Include a summary so the customer need not repeat it.",
        "inputSchema": {"json": {"type": "object",
                                 "properties": {"reason": {"type": "string"},
                                                "summary": {"type": "string"}},
                                 "required": ["reason", "summary"]}}}},
]


def lookup_order(order_id: str) -> dict:
    order = ORDERS.get(order_id.strip().upper().lstrip("#"))
    if not order:
        return {"error": f"No order found with ID {order_id}"}
    return {"order_id": order_id.strip().upper().lstrip("#"), **order}


def initiate_return(order_id: str, reason: str) -> dict:
    order = ORDERS.get(order_id.strip().upper().lstrip("#"))
    if not order:
        return {"approved": False, "error": f"No order found with ID {order_id}"}
    if order["final_sale"]:
        return {"approved": False, "error": "Final sale merchandise cannot be returned (POL-RET-004 §4). "
                                            "Exceptions require the Director of Customer Experience."}
    if not order["delivered"]:
        return {"approved": False, "error": "Order has not been delivered yet."}
    days = (TODAY - date.fromisoformat(order["delivered"])).days
    window = WINDOW_DAYS[order["tier_at_purchase"]]
    if days > window:
        return {"approved": False, "error": f"Outside the {window}-day window ({days} days since delivery). "
                                            "Window exceptions need store staff or a human agent."}
    return {"approved": True, "return_id": f"RET-{order_id[-6:]}", "days_since_delivery": days,
            "window_days": window}


def apply_loyalty_credit(customer_id: str, amount: float, reason: str) -> dict:
    if amount <= 0 or amount > AGENT_CREDIT_LIMIT:
        return {"applied": False, "error": f"Amount must be between $0 and ${AGENT_CREDIT_LIMIT:.0f}; "
                                           "larger amounts need a human agent."}
    return {"applied": True, "customer_id": customer_id, "amount": amount}


def escalate_to_human(reason: str, summary: str) -> dict:
    return {"escalated": True, "ticket_id": "ESC-" + str(abs(hash(summary)) % 100000).zfill(5),
            "message": "A human specialist will follow up within one business day."}
