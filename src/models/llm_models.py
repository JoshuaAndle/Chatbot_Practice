
import os
import time


import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import deque

class BaseChatBot:
    def __init__(self, max_history):
        self.max_history = max_history

        self.chat_history = deque([]) # Using a queue where each element is one prompt-answer pair

        self.model = None
        self.tokenizer = None

    ### Inherited by child classes to parse outputs as needed based on their output format
    def parse_output(self, model_inputs, generated_ids):
        pass




    def clear_history(self):
        ### Resets the chat history for starting a new conversation
        self.chat_history = deque([])



    ### Inherited function for applying retrieved documents as context. Handled by child classes
    def append_documents(self, query, scores, documents):
        if documents is None:
            return query

        #!# Eventually I may want to only include top-p documents, but I might just handle that at the retrieval stage.
        #!#    Either way I am sticking with just applying all documents for the time being. 
        document_prompt = f"""
        When answering this prompt use the following papers when they are useful:
        {documents}

        Answer this prompt:
        {query}"""

        return document_prompt

    def get_history(self):
        context = []
        if len(self.chat_history) > 0:
            for exchange in range(len(self.chat_history)):
                context.extend(self.chat_history[exchange])
            print("\nContext of past chat window: \n", context)
        return context


    def apply_context(self, query, scores=None, documents=None):
        context = self.get_history() # Apply chat history context
        prompt = self.append_documents(query, scores, documents) # Apply document context

        ### prepare the model input using the chat template
        message = [{"role": "user", "content": prompt}]
        context.extend(message)
        return context, message


    def prompt(self, query, scores=None, documents=None):
        
        context_query, message = self.apply_context(query, scores, documents)
        self.chat_history.append(message)

        text = self.tokenizer.apply_chat_template(
            context_query,
            tokenize=False,
            add_generation_prompt=True,
        )



        ### Tokenize final prompts
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        ### Get model response
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1000
        )

        ### Parse response as needed and return result
        print("Full Prompt: \n", text)

        response = self.parse_output(model_inputs, generated_ids)

        print("-"*120, "\n\n\nResponse:", response)

        message = [{"role": "assistant", "content": response}]
        self.chat_history.append(message)

        ### If chat context is too long, remove the oldest prompt-response pair
        if len(self.chat_history) > self.max_history:
            self.chat_history.popleft()
            self.chat_history.popleft()
            








class QwenReasoning(BaseChatBot):
    ### Chatbot built on Qwen3-0.6B reasoning model
    def __init__(self, max_history):
        super().__init__(max_history)

        model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )



    def parse_output(self, model_inputs, generated_ids):
        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

        # parsing thinking content
        try:
            # rindex finding 151668 (</think>)
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        thinking_content = self.tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
        content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
        print("thinking content:", thinking_content)

        return content





class QwenInstruct(BaseChatBot):
    ### Chatbot built on Qwen2.5-0.5B Instruct model
    def __init__(self, max_history):
        super().__init__(max_history)

        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )


    def parse_output(self, model_inputs, generated_ids):
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response







