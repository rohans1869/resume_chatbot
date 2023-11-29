"""## RetrievalQA with LLaMA 2-70B on Together API"""
# import libraries
import os
import together
import logging
from typing import Any, Dict, List, Mapping, Optional
from pydantic import Extra, Field, root_validator
from langchain.llms.base import LLM
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.chains.question_answering import load_qa_chain
import gradio as gr

# set your API key
os.environ["TOGETHER_API_KEY"] = "6216ce36aadcb06c35436e7d6bbbc18b354d8140f6e805db485d70ecff4481d0"
together.api_key = os.environ["TOGETHER_API_KEY"]

# set llama model
together.Models.start("togethercomputer/llama-2-70b-chat")


class TogetherLLM(LLM):
    """Together large language models."""

    model: str = "togethercomputer/llama-2-70b-chat"
    """model endpoint to use"""

    together_api_key: str = os.environ["TOGETHER_API_KEY"]
    """Together API key"""

    temperature: float = 0.7
    """What sampling temperature to use."""

    max_tokens: int = 512
    """The maximum number of tokens to generate in the completion."""

    class Config:
        extra = Extra.forbid

    @property
    def _llm_type(self) -> str:
        """Return type of LLM."""
        return "together"

    def _call(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> str:
        """Call to Together endpoint."""
        together.api_key = self.together_api_key
        output = together.Complete.create(prompt,
                                          model=self.model,
                                          max_tokens=self.max_tokens,
                                          temperature=self.temperature,
                                          )
        text = output['output']['choices'][0]['text']
        return text

# Load and process the text files
loader = TextLoader('resume_data.txt')
# loader = DirectoryLoader('./folder/', glob="./*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

# Make a chain
llm = TogetherLLM(
    model= "togethercomputer/llama-2-70b-chat",
    temperature = 0.1,
    max_tokens = 1024
)

# chain
chain = load_qa_chain(llm=llm, chain_type="stuff")
query1= "what is this story about?"
chain.run(input_documents=documents, question=query1)


# gradio
description = "This is a chatbot application based on the llama2 70B model. Simply type an input to get started with chatting."
examples = [["what is your contact number?"], ["where you are currently working?"]]


def greet(query1, history):
  return chain.run(input_documents=documents, question="answer as if person responding. do not ask question back. \n Question: "+query1)

gr.ChatInterface(greet,title = "Chat with my Bot", description=description,examples=examples).launch(debug = True)