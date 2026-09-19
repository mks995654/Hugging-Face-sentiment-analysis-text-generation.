from HuggingFace import predict_sentiment

def test_positive_sentiment():
    result = predict_sentiment("This is one of the best experiences I have ever had.")
    assert result == "POSITIVE"
    print("Positive sentiment test passed.")

def test_negative_sentiment():
    result = predict_sentiment("The service was terrible and extremely disappointing.")
    assert result == "NEGATIVE"
    print("Negative sentiment test passed.")

def test_neutral_sentiment():
    result = predict_sentiment("I am not sure if i like this product or not")
    assert result in ["POSITIVE", "NEGATIVE"]  # The model may classify neutral sentiment as either positive or negative.
    print("Neutral sentiment test passed.")

from HuggingFace import generate_text

def test_text_generation():
    result = generate_text("Once upon a time")
    assert isinstance(result, str)
    print("result:", result)
    print("Text generation test passed.")