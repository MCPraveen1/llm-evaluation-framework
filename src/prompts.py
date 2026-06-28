SENTIMENT_PROMPT = """
You are a sentiment analysis expert.

Analyze the following text.

Text:
{text}

Return ONLY one word.

positive

or

negative

Do not explain.
Do not add punctuation.
"""