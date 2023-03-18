import json
import os

SPICE = [
	"Include technical detail if possible.",
	"Do not begin with \"As an AI language model, ...\"",
]

SUMMARY_SYS_PROMPT = "You are a helpful assistant who can summarize a user-provided page of text with information of previous pages in mind.\
		Note that you may or may not need to use the information from previous sections."
SUMMARY_USER_PROMPT = "Summarize this text snippet with fewer than 200 words, based on the summary from the previous sections.\
begin with \"the section ...\" and include technical details if possible."

FINAL_SUMMARY_PROMPT = "Summarize with detail. Please include the name of the paper at the beginning."

INTERESTING_PROMPT = "List two interesting things about the paper. Things that might interest general readers with technical background.\
Begin with something similiar to but not the same as \" I find it interesting because...\". Each thing should have its own paragraph."

DISLIKE_PROMPT = "Hypothetically, what is the one thing general readers might be confused about the paper, or what is the one thing that is left unexplained?"

QUESTION_PROMPT = "Hypothetically, what are the two possible technical questions that another research fellow may ask the author about the topic presented in this paper?\
List the two questions only and do not answer them."

IMITATION_PROMPT_FMT = "Given a writing sample:\n{}\nImitate this style, re-write the following paragraph:\n{}"

WRITING_SAMPLE = 'writing_sample'

class SummaryAlgorithm:
	FULL_CONTEXT : int = 0
	NAIVE : int = 1

class QAAlgorithm:
	FULL_CONTEXT : int = 0
	NAIVE : int = 1

API_KEY_FILE = './key.txt'
CONFIG_FILE = './config.json'

def load_last_api_key():
	if os.path.exists(API_KEY_FILE):
		with open(API_KEY_FILE, 'r') as f:
			return f.readline()
def set_api_key(val : str):
	with open(API_KEY_FILE, 'w') as f:
		f.write(val)

if os.path.exists(CONFIG_FILE):
	with open(CONFIG_FILE, 'r', encoding = 'utf-8') as f:
		data = json.load(f)
else:
	data = {}

def get(name : str):
	if name in data:
		return data[name]
	else:
		return ''

def set(name : str, val):
	data[name] = val

def get_imitation_prompt(sample : str, text : str) -> str:
	return IMITATION_PROMPT_FMT.format(sample, text)

def save():
	with open(CONFIG_FILE, 'w', encoding = 'utf-8') as f:
		json.dump(data, f)
if __name__ == "__main__":
	print(data)
	save()