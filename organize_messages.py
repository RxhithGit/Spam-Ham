def categorize_message(message: str) -> str:
    lower_msg = message.lower()
    if "bank" in lower_msg or "transaction" in lower_msg:
        return "bank"
    elif "social" in lower_msg or "facebook" in lower_msg or "instagram" in lower_msg:
        return "social"
    else:
        return "other"
