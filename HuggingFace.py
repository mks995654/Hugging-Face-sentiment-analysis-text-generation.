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

# Sample text
text = "She is not coming to the meeting today."

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")  # The return_tensors="pt" argument specifies that the output should be in the form of PyTorch tensors, which are suitable for input to the model.

print(inputs)

with torch.no_grad(): # As using pretrained model, we don't need to compute gradients. Weight updates and loss calculations are not required.
    outputs = model(**inputs) # model takes input in form of a dictionary,
    # where the keys are the names of the input parameters and the values are the corresponding tensors. 
    # The ** operator is used to unpack the dictionary into keyword arguments for the model's forward method.
    
# Another method is input_ids, and attention_mask is their own parameter.
# model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']) # pass both parameters seprately.

# get the logits.
# The logits are the raw numbers , unnormalized predictions of the model. Let's extract the logits from the model's outputs to perform further processing, such as determining the predicted class or calculating probabilities.

logits = outputs.logits
print(logits.shape) # input shape is [batch_size, num_labels] = [1, 2] for binary classification.

# Convert logits to probabilities
probs = torch.softmax(logits, dim=-1) # Softmax convert Scores into probabilities.

# Get the predicted class
predicted_class = torch.argmax(probs, dim=-1) # Argmax returns the index of the maximum value along a specified dimension.

# Map the predicted class to the label
labels = ["NEGATIVE", "POSITIVE"]
predicted_label = labels[predicted_class]

print(f"Predicted label: {predicted_label}")


# Text Generation with GPT-2

tokenizer2 = GPT2Tokenizer.from_pretrained('gpt2') # Load GPT-2 tokenizer. The GPT-2 tokenizer is responsible for converting text into a format that the GPT-2 model can understand, and vice versa.
model2 = GPT2LMHeadModel.from_pretrained('gpt2') # Load GPT-2 model. The GPT-2 model is a large-scale transformer-based language model that can generate coherent and contextually relevant text based on the input it receives.

# Set padding token
tokenizer2.pad_token = tokenizer2.eos_token

# Tokenize input text
inputs2 = tokenizer2(
    "Once upon a time",
    return_tensors="pt"
)

with torch.no_grad():
    output_ids = model2.generate(
        input_ids=inputs2["input_ids"],
        attention_mask=inputs2["attention_mask"],
        pad_token_id=tokenizer2.eos_token_id,
        max_length=50,
        num_return_sequences=1
    )

# Convert token IDs back to text
generated_texts = [
    tokenizer2.decode(output, skip_special_tokens=True)
    for output in output_ids
]

for i, text in enumerate(generated_texts):
    print(f"Generated text {i+1}: {text}")