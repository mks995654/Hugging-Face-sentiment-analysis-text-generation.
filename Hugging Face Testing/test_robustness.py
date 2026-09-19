from HuggingFace import predict_sentiment

def test_robust_positive_sentiment():

    input_texts = [
        "I love this product.",
        "I love this product! 😊",
        "I love this product! 😍",
        "I love this product! ❤️"]
    for text in input_texts:
        result = predict_sentiment(text)
        assert result == "POSITIVE"
        print(f"Positive sentiment test passed for input: {text}")
    
def test_robust_negative_sentiment():
    input_texts = [
        "I don't like this product.",
        "I really dislike this product.",
        "I hate this product.",
        "This product is terrible.",
        "I'm very disappointed with this product.",
        "This is an awful product."
    ]
    for text in input_texts:
        result = predict_sentiment(text)
        assert result == "NEGATIVE"
        print(f"Negative sentiment test passed for input: {text}")

def test_robust_neutral_sentiment():
    input_texts = [
        "The meeting is scheduled for Monday",
        "The meeting will take place on Monday",
        "A meeting has been planned for Monday"
    ]
    for text in input_texts:
        result = predict_sentiment(text)
        assert result in ["POSITIVE", "NEGATIVE"]  # The model may classify neutral sentiment as either positive or negative.
        print(f"Neutral sentiment test passed for input: {text}")

from HuggingFace import generate_text

def test_robust_text_generation():
    prompts = [
        "The meeting is scheduled for Monday and"
        "The meeting will take place on Monday and"
        "A meeting has been planned for Monday and"]

    for prompt in prompts:
        result = generate_text(prompt)
        assert isinstance(result, str)
        print(f"result for prompt '{prompt}':", result)
        print("Text generation test passed.")
    