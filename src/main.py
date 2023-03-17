import os
from json import load
from queue import Queue

import openai
import PyPDF2
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QThread, pyqtSignal

from prompt import Prompt

API_KEY_FILE = './key.txt'
def load_last_api_key():
	if os.path.exists(API_KEY_FILE):
		with open(API_KEY_FILE, 'r') as f:
			return f.readline()
DEFAULT_API_KEY = load_last_api_key()

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

class ResultSectionType:
	SUMMARY : int = 0
	INTERESTING : int = 1
	DISLIKE : int = 2
	QUESTION : int = 3

	@staticmethod
	def type_to_str(type):
		if type == ResultSectionType.SUMMARY:
			return 'SUMMARY'
		elif type == ResultSectionType.INTERESTING:
			return 'INTERESTING'
		elif type == ResultSectionType.DISLIKE:
			return 'DISLIKE'
		elif type == ResultSectionType.QUESTION:
			return 'QUESTION'
		raise RuntimeError('unknown result section type')
	
def get_result_types() -> list[str]:
	ret = []
	for attr_name, attr_value in ResultSectionType.__dict__.items():
		if not callable(attr_value):
			if not attr_name.startswith('__') or not attr_name.endswith('__'):
				ret.append(attr_name)
	return ret

class SummaryAlgorithm:
	FULL_CONTEXT = 0
	NAIVE = 1

class QAAlgorithm:
	FULL_CONTEXT = 0
	NAIVE = 1

def get_result_section_prompt(page_summary : list[str], type : int) -> Prompt:
	p = Prompt()
	if type == ResultSectionType.SUMMARY:
		# p.add(Prompt.SYS).add()
		user = p.add(Prompt.USER).add_important(FINAL_SUMMARY_PROMPT + '\n')
		for summary in page_summary:
			user.add(summary)
	else:
		assist = p.add(Prompt.ASSIST).add_important("the summary of the paper is: \n")
		for summary in page_summary:
			assist.add(summary)
		for spice in SPICE:
			assist = p.add(Prompt.ASSIST).add(spice)

		user = (p.add(Prompt.USER)
				.add_important("please answer the following question based on the summary of the academic paper provided above"))
		if type == ResultSectionType.INTERESTING:
			user.add(INTERESTING_PROMPT)
		elif type == ResultSectionType.DISLIKE:
			user.add(DISLIKE_PROMPT)
		elif type == ResultSectionType.QUESTION:
			user.add(QUESTION_PROMPT)
	return p

class WorkerResult(object):
	def __init__(self):
		super().__init__()
		self.paper_section_summary : list[str] = []
		self.results : dict[str,str] = {}
	
	def set_section(self, name : str, text : str):
		self.results[name] = text
	
	def get_section(self, name : str) -> str:
		if name not in self.results:
			raise RuntimeError('unknown section: {}'.format(name))
		return self.results[name]
	
	def write_plain(self, file):
		for _, text in self.results.items():
			file.write(text)
			file.write('-' * 10)
			file.write('\n')

class PdfWorker(QThread):
	PROCESS_REQUEST = 0
	REDO_REQUEST = 1
	TERMINATE_REQUEST = 2
	
	progress_signal = pyqtSignal(str, int, int)
	result_receiver_signal = pyqtSignal(WorkerResult)

	def __init__(self, parent, pdf_name : str, summary_algorithm : int, qa_algorithm : int):
		super().__init__(parent)
		self.pdf_name = pdf_name
		self.summary_algorithm = summary_algorithm
		self.qa_algorithm = qa_algorithm
		self.request_queue = Queue()

	def update_prog(self, msg = ''):
		self.cur_prog = min(self.cur_prog + 1, self.total_prog)
		self.progress_signal.emit(msg, self.cur_prog, self.total_prog)

	def process_sections(self, pages) -> list[str]:
		p = Prompt()
		p.add(Prompt.SYS).add(SUMMARY_SYS_PROMPT)

		page_summary = []

		user = None # stores user message
		for i, page in enumerate(pages):
			# remove previous user query
			p.remove(user)

			# formulate user prompt
			user = p.add(Prompt.USER)
			(user.add_important(SUMMARY_USER_PROMPT + '\n')
				.add(page.extract_text()))

			cur_page_summary = p.dispatch()
			if self.summary_algorithm == SummaryAlgorithm.FULL_CONTEXT:
				# store the current summary as context
				# later pages have greater importance
				context = p.add(Prompt.ASSIST)
				(context.add_important('the summary of page {} is:\n'.format(i))
					.add(cur_page_summary, i))

			self.update_prog('processing page {}'.format(i))
			page_summary.append(cur_page_summary)
		return page_summary

	def run(self):
		result_section_types = get_result_types()

		while True:
			task_type, arg = self.request_queue.get()
			if task_type == PdfWorker.PROCESS_REQUEST:
				self.result = WorkerResult()

				self.cur_prog = 0
				reader = PyPDF2.PdfFileReader(self.pdf_name)
				self.total_prog = len(reader.pages) + len(result_section_types) + 1

				with open('out.txt', 'w', encoding="utf-8") as text_file:
					self.result.paper_section_summary = self.process_sections(reader.pages)
					all_summary = self.result.paper_section_summary
					# text_file.write(str(section_summary))
					# text_file.write('\n\n')

					if self.qa_algorithm == QAAlgorithm.FULL_CONTEXT:
						for i, type_name in enumerate(result_section_types):
							result = get_result_section_prompt(all_summary, i).dispatch()
							self.result.set_section(type_name, result)
							self.update_prog('writing section {}'.format(type_name))
					else:
						# naive approach uses the total summary as context for answering questions
						type = ResultSectionType.SUMMARY
						total_summary = get_result_section_prompt(all_summary, type).dispatch()
						self.result.set_section(result_section_types[type], total_summary)
						self.update_prog('writing section {}'.format(result_section_types[type]))

						for i, type_name in enumerate(result_section_types):
							if i == ResultSectionType.SUMMARY:
								continue
							self.result.set_section(type_name, 
								get_result_section_prompt([total_summary], i).dispatch())
							self.update_prog('writing section {}'.format(type_name))

					self.result.write_plain(text_file)
					self.update_prog()
			elif task_type == PdfWorker.REDO_REQUEST:
				redo_type, = arg
				self.cur_prog, self.total_prog = 0, 2
				all_summary = self.result.paper_section_summary
				self.update_prog('rewriting section {}'.format(result_section_types[redo_type]))

				if self.qa_algorithm == QAAlgorithm.FULL_CONTEXT:
					text = get_result_section_prompt(all_summary, redo_type).dispatch()
					self.result.set_section(result_section_types[redo_type], text)
				else:
					if redo_type == ResultSectionType.SUMMARY:
						total_summary = get_result_section_prompt(all_summary, redo_type).dispatch()
						self.result.set_section(result_section_types[redo_type], total_summary)
					else:
						total_summary = self.result.get_section(result_section_types[ResultSectionType.SUMMARY])
						text = get_result_section_prompt(all_summary, redo_type).dispatch()
						self.result.set_section(result_section_types[redo_type], text)
				with open('out.txt', 'w', encoding="utf-8") as text_file:
					self.result.write_plain(text_file)
				self.update_prog()

			elif task_type == PdfWorker.TERMINATE_REQUEST:
				break
			else:
				raise RuntimeError('unknown worker request: {}'.format(task_type))

class Window(QtWidgets.QMainWindow):
	def __init__(self):
		super().__init__()
		self.init_ui()
		openai.api_key = DEFAULT_API_KEY

		self.pdf_file = ''
		self.summary_algorithm = SummaryAlgorithm.NAIVE
		self.qa_algorithm = QAAlgorithm.NAIVE
		self.worker = None

	def init_ui(self):
		uic.loadUi('window.ui', self)
		self.browseBtn.clicked.connect(self.get_pdf)
		self.processBtn.clicked.connect(self.process_pdf)
		self.processBtn.setEnabled(False)
		self.contextSummaryBtn.stateChanged.connect(self.set_summary_algorithm)
		self.contextSummaryBtn.setCheckState(0)
		self.fullContextQABtn.stateChanged.connect(self.set_qa_algorithm)
		self.fullContextQABtn.setCheckState(0)
		self.redoBtn1.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.SUMMARY))
		self.redoBtn2.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.INTERESTING))
		self.redoBtn3.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.DISLIKE))
		self.redoBtn4.clicked.connect(lambda : self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.QUESTION))
		self.paraphraseBtn.clicked.connect(self.redo_all)
		self.apiKeyText.setText(DEFAULT_API_KEY)
		self.apiKeyText.editingFinished.connect(self.set_api_key)
		self.set_pdf_dependent_btns(False)
		self.show()

	def get_pdf(self):
		fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open PDF', '', 'PDF Files (*.pdf)')
		# check if the file exists
		if os.path.exists(fname[0]):
			self.pdf_file = fname[0]
			self.processBtn.setEnabled(True)
		else:
			self.pdf_file = ''
			self.processBtn.setEnabled(False)
			self.print("file not found")
	
	def set_pdf_dependent_btns(self, state : bool):
		self.redoBtn1.setEnabled(state)
		self.redoBtn2.setEnabled(state)
		self.redoBtn3.setEnabled(state)
		self.redoBtn4.setEnabled(state)
		self.paraphraseBtn.setEnabled(state)

	def send_worker_request(self, type, *args):
		if self.worker is not None:
			self.worker.request_queue.put((type, args))
	def new_worker(self):
		self.worker = PdfWorker(self, self.pdf_file, self.summary_algorithm, self.qa_algorithm)
		self.worker.progress_signal.connect(self.set_progress)
		self.worker.start()
	def redo_all(self):
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.SUMMARY)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.INTERESTING)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.DISLIKE)
		self.send_worker_request(PdfWorker.REDO_REQUEST, ResultSectionType.QUESTION)
	def process_pdf(self):
		self.send_worker_request(PdfWorker.TERMINATE_REQUEST)
		self.new_worker()
		self.browseBtn.setEnabled(False)
		self.processBtn.setEnabled(False)
		self.set_pdf_dependent_btns(False)
		self.send_worker_request(PdfWorker.PROCESS_REQUEST)

	def set_api_key(self):
		openai.api_key = self.apiKeyText.text()
		with open(API_KEY_FILE, 'w') as f:
			f.write(self.apiKeyText.text())

	def set_summary_algorithm(self, state : int):
		if state > 0:
			self.summary_algorithm = SummaryAlgorithm.FULL_CONTEXT
		else:
			self.summary_algorithm = SummaryAlgorithm.NAIVE

	def set_qa_algorithm(self, state : int):
		if state > 0:
			self.qa_algorithm = QAAlgorithm.FULL_CONTEXT
		else:
			self.qa_algorithm = QAAlgorithm.NAIVE

	def print(self, text):
		self.messageLabel.setText(text)
		self.messageLabel.adjustSize()

	def set_progress(self, msg, value, total):
		# set progress bar
		if value == 0:
			self.pbar.setValue(0)
			self.print('')
		elif value == total:
			self.pbar.setValue(0)
			self.browseBtn.setEnabled(True)
			self.processBtn.setEnabled(True)
			self.set_pdf_dependent_btns(True)
			self.print('finished')
		else:
			self.pbar.setValue(int(value / total * 100))
			self.print(msg)

app = QtWidgets.QApplication([])
a_window = Window()
app.exec_()