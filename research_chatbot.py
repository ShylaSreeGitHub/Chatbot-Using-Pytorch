import random  # To randomly choose from the possible answers
import json

import torch

from feed_forward_neural_network import NeuralNetwork
from preprocesing_data_nltk_utils import bag_of_words, tokenize  # All the text preprocessing functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # To check if we have the GPU support

# To Open the dataset in READ mode
with open(r'research_dataset.json', 'r') as f:   
    intents = json.load(f)

# Open and Load the Saved file
FILE = "data_model.pth"
data = torch.load(FILE)    

#print("***********\n",data)


input_layer_size = data["input_layer_size"]
hidden_layer_size = data["hidden_layer_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNetwork(input_layer_size, hidden_layer_size, output_size).to(device)  
model.load_state_dict(model_state)

model.eval() # Set to Evaluation mode

bot_name = "ScrapT" # Name of our Chatbot

def get_response(msg):
    #sentence = input("You: ")
    #if sentence == "bye":
     #   print("{}:Bye! Take care!".format(bot_name))
      #  break

    user_chat = tokenize(msg) # To Tokenize the sentence 
    X = bag_of_words(user_chat, all_words) # Returns a Numpy array
    X = X.reshape(1, X.shape[0]) # Reshape the bag of words structure in Rows and Columns
    X = torch.from_numpy(X).to(device) # Convert in to a Torch Tensor

    output = model(X) # Gives us predictions
    _, predicted = torch.max(output, dim=1)

    # Softmax is used to get the probabilities of each Output class
    tag = tags[predicted.item()] # Tags present on our Dataset

   
    probs = torch.softmax(output, dim=1)  # Softmax or Probability of Output class
    prob = probs[0][predicted.item()] # Probability for the Predicted tag

     # To check if the Probabilities of the 'tag' is high enough to be mapped (Probability should be greater than 75%)
    if prob.item() > 0.75: 
        for intent in intents['intents']: # Loop through the dataset
            if tag == intent["tag"]: # Check for the specific tag in the dataset
                return random.choice(intent['responses']) # Print a Response 

    return "I did not quite catch your intention... Please drop your email id, our team will get back to you!"



