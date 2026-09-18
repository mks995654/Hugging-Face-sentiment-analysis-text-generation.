from transformers import pipeline    # transformer pipeline a high level API for using pre-trained models for various NLP tasks.
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer # Pre trained tokenizer and model for text generation using GPT-2, a popular language model developed by OpenAI.
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer # Pre trained tokenizer and model for sentiment analysis using DistilBERT, a smaller and faster version of BERT.

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

# Sentiment Analysis with DistilBERT

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english', num_labels=2) # num_labels=2 indicates that the model is fine-tuned for binary classification (positive or negative sentiment).

def predict_sentiment(text):

    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")  # The return_tensors="pt" argument specifies that the output should be in the form of PyTorch tensors, which are suitable for input to the model.

    # print(inputs)

    with torch.no_grad():  # As using pretrained model, we don't need to compute gradients. Weight updates and loss calculations are not required.
        outputs = model(**inputs)  # model takes input in form of a dictionary,
        # where the keys are the names of the input parameters and the values are the corresponding tensors.
        # The ** operator is used to unpack the dictionary into keyword arguments for the model's forward method.

        # Another method is input_ids, and attention_mask is their own parameter.
        # model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']) # pass both parameters separately.

        # get the logits.
        # The logits are the raw numbers, unnormalized predictions of the model. Let's extract the logits from the model's outputs to perform further processing, such as determining the predicted class or calculating probabilities.

        logits = outputs.logits
        # print(logits.shape)  # input shape is [batch_size, num_labels] = [1, 2] for binary classification.

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)  # Softmax convert Scores into probabilities.

        # Get the predicted class
        predicted_class = torch.argmax(probs, dim=-1)  # Argmax returns the index of the maximum value along a specified dimension.

        # Map the predicted class to the label
        labels = ["NEGATIVE", "POSITIVE"]
        predicted_label = labels[int(predicted_class[0])]

        # print(f"Predicted label: {predicted_label}")

    return predicted_label


# Text Generation with GPT-2

tokenizer2 = GPT2Tokenizer.from_pretrained("gpt2") # GPT-2 tokenizer is used to convert text into a format that the GPT-2 model can understand. It breaks down the input text into smaller units called tokens, which are then converted into numerical representations (token IDs) that the model can process.
model2 = GPT2LMHeadModel.from_pretrained("gpt2") # GPT-2 model is a language model that can generate coherent and contextually relevant text based on a given prompt. It is trained on a large corpus of text data and can produce human-like text in various styles and formats.

tokenizer2.pad_token = tokenizer2.eos_token # The pad_token is set to the end-of-sequence (eos) token of the GPT-2 tokenizer. This is done to ensure that when padding is applied to input sequences, it uses the eos token as the padding token. Padding is necessary when dealing with batches of sequences of different lengths, as it allows for uniform input sizes.


def generate_text(prompt, max_length=50):

    inputs = tokenizer2(
        prompt,
        return_tensors="pt"
    ) # input taking two parameters, prompt to send to the model and return_tensors="pt" to specify that the output should be in the form of PyTorch tensors.

    with torch.no_grad(): # As using pretrained model, we don't need to compute gradients. Weight updates and loss calculations are not required.
        output_ids = model2.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer2.eos_token_id,
            max_length=max_length,
            num_return_sequences=1
        )

    generated_text = tokenizer2.decode(
        output_ids[0],
        skip_special_tokens=True
    ) # The decode method of the tokenizer is used to convert the generated token IDs back into human-readable text. The skip_special_tokens=True argument ensures that any special tokens (like padding or end-of-sequence tokens) are removed from the final output, resulting in a clean and coherent generated text.

    return generated_text
