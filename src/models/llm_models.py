
import os
import time


import torch
from torch import nn
import torch.nn.functional as F
import pandas as pd
import numpy as np

from torch import Tensor

from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_models import QwenInstruct, QwenReasoning
from collections import deque

class BaseChatBot:
    def __init__(self, history_context, max_history):
        self.max_history = max_history
        self.history_context = history_context

        self.chat_history = deque([]) # Using a queue where each element is one prompt-answer pair
        self.window_index = 0

        self.model = None
        self.tokenizer = None


    def parse_output(self, model_inputs, generated_ids):
        pass


    def prompt(self, prompt):
        
        context = []
        if len(self.chat_history) > 0:
            for exchange in range(self.window_index, len(self.chat_history)):
                context.extend(self.chat_history[exchange])
            print("\nContext of past chat window: \n", context)

        # prepare the model input using the chat template
        messages = [{"role": "user", "content": prompt}]
        context.extend(messages)

        # print("Final context for text")
        text = self.tokenizer.apply_chat_template(
            context,
            tokenize=False,
            add_generation_prompt=True,
        )
        self.chat_history.append(messages)

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        # conduct text completion
        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=1000
        )


        print("Prompt: \n", text)

        response = self.parse_output(model_inputs, generated_ids)

        print("content:", response)

        messages = [{"role": "assistant", "content": response}]
        self.chat_history.append(messages)

        if len(self.chat_history) > self.max_history:
            self.chat_history.popleft()
            self.chat_history.popleft()
            
        ### Ensures the context window moves as needed while the chat history is growing
        if (len(self.chat_history) - self.window_index) > self.history_context:
            self.window_index += 2


class QwenReasoning(BaseChatBot):
    ### Chatbot built on Qwen3-0.6B reasoning model
    def __init__(self, history_context, max_history):
        super().__init__(history_context, max_history)

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
    def __init__(self, history_context, max_history):
        super().__init__(history_context, max_history)

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







