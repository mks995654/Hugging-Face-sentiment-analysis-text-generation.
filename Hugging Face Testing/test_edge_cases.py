from HuggingFace import predict_sentiment

def test_edge_positive_sentiment():

    input_texts = [
        "GOOD!!!",
        "good good good",
        "I LOVE IT!!! ❤️",
        "Absolutely fantastic!!!",
        "Wow... amazing!",
        "Couldn't be happier!",
        "This is surprisingly good.",
        "Not bad at all.",
        "I can't say enough good things about it.",
        "Best. Purchase. Ever.",
        "😍😍😍"]

    for text in input_texts:
        result = predict_sentiment(text)
        assert result == "POSITIVE"
        print(f"Positive sentiment test passed for input: {text}")
    
def test_edge_negative_sentiment():
    input_texts = [
        "BAD!!!",
        "bad bad bad",
        "I HATE IT!!! 🤬",
        "Terrible experience.",
        "Worst purchase ever.",
        "I am so disappointed.",
        "This is the worst thing I have ever bought.",
        "Not good at all.",
        "I can't stand this product.",
        "I would never recommend this to anyone.",
        "🤮🤮🤮"
    ]
    for text in input_texts:
        result = predict_sentiment(text)
        assert result == "NEGATIVE"
        print(f"Negative sentiment test passed for input: {text}")

def test_edge_neutral_sentiment():
    input_texts = [
        "The meeting is tomorrow.",
        "The package arrived today.",
        "The product costs $50.",
        "I bought two books.",
        "The store opens at 9 AM.",
        "The phone has a 6-inch screen.",
        "My order number is 12345.",
        "The movie starts at 7 PM.",
        "The package contains three items.",
        "The product is available in two sizes."
    ]
    for text in input_texts:
        result = predict_sentiment(text)
        assert result in ["POSITIVE", "NEGATIVE"]  # The model may classify neutral sentiment as either positive or negative.
        print(f"Neutral sentiment test passed for input: {text}")

from HuggingFace import generate_text

def test_edge_text_generation():
    prompts = [
        "I",
        "AI",
        "Hello"]

    for prompt in prompts:
        result = generate_text(prompt)
        assert isinstance(result, str)
        print(f"result for prompt '{prompt}':", result)
        print("Text generation test passed.")
    