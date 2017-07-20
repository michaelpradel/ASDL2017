from os import listdir
from os.path import join, isfile
import json
from random import randint

#########################################
## START of part that students may change
from code_completion_baseline import Code_Completion_Baseline

training_dir = "./../../programs_800/"
query_dir = "./../../programs_200/"

model_file = "./../../trained_model"
use_stored_model = False

max_hole_size = 2
simplify_tokens = True
## END of part that students may change
#########################################

def simplify_token(token):
    if token["type"] == "Identifier":
        token["value"] = "ID"
    elif token["type"] == "String":
        token["value"] = "\"STR\""
    elif token["type"] == "RegularExpression":
        token["value"] = "/REGEXP/"
    elif token["type"] == "Numeric":
        token["value"] = "5"

# load sequences of tokens from files
def load_tokens(token_dir):
    token_files = [join(token_dir, f) for f in listdir(token_dir) if isfile(join(token_dir, f)) and f.endswith("_tokens.json")]
    token_lists = [json.load(open(f, encoding='utf8')) for f in token_files]
    if simplify_tokens:
        for token_list in token_lists:
            for token in token_list:
                simplify_token(token)
    return token_lists

# removes up to max_hole_size tokens
def create_hole(tokens):
    hole_size = min(randint(1, max_hole_size), len(tokens) - 1)
    hole_start_idx = randint(1, len(tokens) - hole_size)
    prefix = tokens[0:hole_start_idx]
    expected = tokens[hole_start_idx:hole_start_idx + hole_size]
    suffix = tokens[hole_start_idx + hole_size:]
    return(prefix, expected, suffix)

# checks if two sequences of tokens are identical
def same_tokens(tokens1, tokens2):
    if len(tokens1) != len(tokens2):
        return False
    for idx, t1 in enumerate(tokens1):
        t2 = tokens2[idx]
        if t1["type"] != t2["type"] or t1["value"] != t2["value"]:
            return False  
    return True

#########################################
## START of part that students may change
code_completion = Code_Completion_Baseline()
## END of part that students may change
#########################################

# train the network
training_token_lists = load_tokens(training_dir)
if use_stored_model:
    code_completion.load(training_token_lists, model_file)
else:
    code_completion.train(training_token_lists, model_file)

# query the network and measure its accuracy
query_token_lists = load_tokens(query_dir)
correct = incorrect = 0
for tokens in query_token_lists:
    (prefix, expected, suffix) = create_hole(tokens)
    completion = code_completion.query(prefix, suffix)
    if same_tokens(completion, expected):
        correct += 1
    else:
        incorrect += 1
accuracy = correct / (correct + incorrect)
print("Accuracy: " + str(correct) + " correct vs. " + str(incorrect) + " incorrect = "  + str(accuracy))

