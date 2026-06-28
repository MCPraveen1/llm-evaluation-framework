def parse_sentiment(text):
    """
    Convert model output to binary label.
    positive -> 1
    negative -> 0
    """

    text = text.lower().strip()

    if text == "positive":
        return 1

    elif text == "negative":
        return 0

    return 0