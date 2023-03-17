import heapq
import typing

import openai
from tiktoken.core import Encoding
from tiktoken.load import load_tiktoken_bpe


def cl100k_base():
	mergeable_ranks = load_tiktoken_bpe(
		"https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
	)
	ENDOFTEXT = "<|endoftext|>"
	FIM_PREFIX = "<|fim_prefix|>"
	FIM_MIDDLE = "<|fim_middle|>"
	FIM_SUFFIX = "<|fim_suffix|>"
	ENDOFPROMPT = "<|endofprompt|>"
	special_tokens = {
		ENDOFTEXT: 100257,
		FIM_PREFIX: 100258,
		FIM_MIDDLE: 100259,
		FIM_SUFFIX: 100260,
		ENDOFPROMPT: 100276,
	}
	return {
		"name": "cl100k_base",
		"pat_str": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
		"mergeable_ranks": mergeable_ranks,
		"special_tokens": special_tokens,
	}


class Message(object):
	"""
	represents the text in the content field
	"""
	def __init__(self, prompt : 'Prompt'):
		self.prompt : 'Prompt' = prompt
		self.important : list[tuple[int, str]] = []
		self.non_important : list[tuple[int, int, int, str]] = []
		self.time_stamp : int = 0 # used to recover original message order

	def add_important(self, text : str) -> 'Message':
		""" 
		adds a piece of text that cannot be deleted nor shortened when token limit is reached
		"""
		self.important.append((self.time_stamp, text))
		self.time_stamp += 1
		return self

	def add(self, text : str, importance : int = 0) -> 'Message':
		""" 
		adds a piece of text that can be deleted, 
		and shortened based on importance and number of times it has been shortened
		when token limit is reached
		"""
		heapq.heappush(self.non_important, (0, importance, self.time_stamp, text))
		self.time_stamp += 1
		return self

	def shorten(self) -> bool:
		if len(self.non_important) > 0:
			cnt, importance, time_stamp, text = heapq.heappop(self.non_important)
			if cnt < 3: # just delete it if we have shortened it for too many num of times
				text = self.prompt._summarize(text)
				heapq.heappush(self.non_important, (cnt+1, importance, time_stamp, text))
			return True
		else:
			return False

	def get_text(self) -> str:
		texts = []
		for (time, text) in self.important:
			texts.append((time, text))
		for (_, _, time, text) in self.non_important:
			texts.append((time, text))
		texts.sort()
		return ''.join(t[1] for t in texts)
	
	def __repr__(self) -> str:
		return self.get_text()

class Prompt(object):

	ASSIST = "assistant"
	SYS = "system"
	USER = "user"

	def __init__(self, limit = 4000):
		super(Prompt, self).__init__()
		self.limit : int = limit
		# list of pairs of (role, message object)
		self.messages : list[tuple[str, Message]] = []
		self.encoding = Encoding(**cl100k_base())

	def _get_num_tokens(self) -> int:
		ret = 0
		for _, msg in self.messages:
			ret += len(self.encoding.encode(msg.get_text()))
		# print('cur num of tokens = {}'.format(ret))
		# print(self.messages)
		return ret

	def add(self, role : str) -> Message:
		msg = Message(self)
		self.messages.append((role, msg))
		return msg

	def remove(self, in_msg : Message | None) -> bool:
		if in_msg is None:
			return False
		for i, (_, msg) in enumerate(self.messages):
			if msg is in_msg:
				del self.messages[i]
				return True
		return False

	def _request(self, messages : list[dict[str,str]], num_tries : int = 10) -> str:
		for i in range(num_tries):
			try:
				completion = openai.ChatCompletion.create(
					model = 'gpt-3.5-turbo', 
					messages = messages
				)
				return completion['choices'][0]['message']['content']
			except BaseException as e:
				if i == num_tries - 1:
					raise e
				else:
					print("request to OpenAI failed, retrying ...")
					continue

	def _summarize(self, text : str):
		messages = [
			{
				"role" : self.SYS,
				"content" : "you are a helpful assistant who is good at summarizing things while retaining as much detail as possible."
			},
			{
				"role" : self.USER,
				"content" : "shorten the following by summarizing concisely:\n" + text
			}
		]
		return self._request(messages)

	def dispatch(self) -> str:
		# make sure we stay within token limit
		while self._get_num_tokens() > self.limit:
			shortened = False
			for role, msg in self.messages:
				if msg.shorten():
					shortened = True
					if self._get_num_tokens() <= self.limit:
						break
			if not shortened:
				raise RuntimeError("no message can be shortened further.")
		message = []
		for role, msg in self.messages:
			message.append({ "role" : role, "content" : msg.get_text() })
		# print('final message {}:\n'.format(message))
		return self._request(message)

def unit_test():
	p = Prompt(50)
	p.add(p.SYS).add_important("You are a helpful assistant")
	p.add(p.USER)\
		.add_important("summarize the following texts:")\
		.add("What is the difference between the list methods append and extend? ", 0)\
		.add("extend iterates over its argument adding each element to the list, extending the list. The length of the list will increase by however many elements were in the iterable argument.", 1)\
		.add("The list.append method appends an object to the end of the list.", 1)\
		.add("Whatever the object is, whether a number, a string, another list, or something else, it gets added onto the end of my_list as a single entry on the list.", 1)
	print(p.dispatch())
if __name__ == "__main__":
	unit_test()