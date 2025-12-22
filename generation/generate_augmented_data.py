'''
Rewrite all queries in input_query_path 
'''
import os
import argparse
import tkinter as tk
import multiprocessing
from tkinter import simpledialog
from tqdm import tqdm
import json
import re

import nltk
nltk.download('words')
nltk.download('wordnet')
from nltk.corpus import words
from nltk.stem import WordNetLemmatizer

# Initialize the WordNet lemmatizer
lemmatizer = WordNetLemmatizer()


from llm import generate_with_multi_gpu_vllm

def is_english_word(word):
    base_form = lemmatizer.lemmatize(word.lower())
    return base_form in words.words()

def remove_leading_number(sentence):
    return re.sub(r'^\d+\.\s*', '', sentence)

class GRF_Prompter:
    def __init__(
        self, 
        phi=10,
        ) -> None:
        
        self.phi = phi 

        # head_instruction
        self.instruction = f"# You will be given an information seeking dialogue between an user and an assistant. Your task: For the user's last question, please provide {phi} different informative responses to answer it."

    
    
    def build_turn_prompt(self, turn):
        this_prompt = [self.instruction]
        
        # current turn

        # previous turn context
        this_dialog = ["# Here is the dialogue:\n"]
        context = turn["ctx_utts_text"]
        if len(context) == 0:
            this_dialog.append("N/A (this is the first question in the dialog, so no previous dialog context)")
        else:
            assert len(context) % 2 == 0, "context length should be even"

            # for every 2 elements in context, they are a pair of user and system turn
            for i in range(0, len(context), 2):
                current_utterance = context[i]
                current_response = context[i+1]
                this_dialog.append(f"user: {current_utterance}\nsystem: {current_response}")
        
        # current turn

        this_dialog.append("# User's last question: " + turn["cur_utt_text"])
        this_dialog = "\n".join(this_dialog)  

        this_prompt.append(this_dialog)
         
        this_prompt.append(f"# Now give me the {self.phi} **different** **informative** responses that answers the users' last question. The format should be:\nresponse1\nresponse2\nresponse3\n....\nresponse{self.phi}\n\n. Go ahead!")
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    

    def parse_returned_text(self, text):

        try:
            splits = text.split('\n')
            result_list = []

            for i in range(len(splits)):
                if splits[i].startswith("#"):
                    splits[i] = splits[i][1:]
                if splits[i].startswith("-"):
                    splits[i] = splits[i][1:]
                if splits[i] == "\n":
                    continue
                if len(splits[i]) < 10:
                    continue
                if ":" in splits[i]:
                    continue
                
                result_list.append(remove_leading_number(splits[i]).strip())

            return result_list

        except Exception as e:
            print(e)

class QR_Prompter:
    def __init__(
        self, 
        phi=10,
        enable_context = False
        ) -> None:
        
        self.enable_context = enable_context
        self.phi = phi 
        # head_instruction

        context_1 = " and the conversation context of the question in an information seeking dialogue" if self.enable_context else "" 

        self.instruction = f"# You will be given a user question{context_1}. please provide {phi} equivalent questions, such that each of the {phi} questions has the same meaning but is in a different form. Write each query on one line."

    
    
    def build_turn_prompt(self, turn):
        
        this_prompt = [self.instruction]
        

        # previous turn context
        this_dialog = ["# Here is the dialogue:\n"]
        context = turn["ctx_utts_text"]

        if len(context) == 0:
            this_dialog.append("N/A (this is the first question in the dialog, so no previous dialog context)")
        else:
            assert len(context) % 2 == 0, "context length should be even"

            # for every 2 elements in context, they are a pair of user and system turn
            for i in range(0, len(context), 2):
                current_utterance = context[i]
                current_response = context[i+1]
                this_dialog.append(f"user: {current_utterance}\nsystem: {current_response}")
        
        # current turn
        dialog = "\n".join(this_dialog) 


        this_prompt.append(dialog)
        this_prompt.append("# Here is the User Question: " + turn["cur_utt_text"])
        
        this_prompt.append(f"# Now give me the {self.phi} different questions. Don't say any other words. Don't generate sequence number of indicator. Just write questions. each question on one line.")
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    

    def parse_returned_text(self, text):

        text = text.strip()

        try:
            splits = text.split('\n')
            result_list = []

            for i in range(len(splits)):
                if splits[i].startswith("#"):
                    splits[i] = splits[i][1:]
                if splits[i] == "\n":
                    continue
                
                result_list.append(remove_leading_number(splits[i]).strip())

            return result_list
        except Exception as e:
            print(e)


class QR_Prompter:
    def __init__(
        self, 
        phi=10,
        enable_context = False
        ) -> None:
        
        self.enable_context = enable_context
        self.phi = phi 
        # head_instruction

        context_1 = " and the conversation context of the question in an information seeking dialogue" if self.enable_context else "" 

        self.instruction = f"# You will be given a user question{context_1}. please provide {phi} equivalent questions, such that each of the {phi} questions has the same meaning but is in a different form. Write each query on one line."

    
    
    def build_turn_prompt(self, turn):
        
        this_prompt = [self.instruction]
        

        # previous turn context
        this_dialog = ["# Here is the dialogue:\n"]
        context = turn["ctx_utts_text"]

        if len(context) == 0:
            this_dialog.append("N/A (this is the first question in the dialog, so no previous dialog context)")
        else:
            assert len(context) % 2 == 0, "context length should be even"

            # for every 2 elements in context, they are a pair of user and system turn
            for i in range(0, len(context), 2):
                current_utterance = context[i]
                current_response = context[i+1]
                this_dialog.append(f"user: {current_utterance}\nsystem: {current_response}")
        
        # current turn
        dialog = "\n".join(this_dialog) 


        this_prompt.append(dialog)
        this_prompt.append("# Here is the User Question: " + turn["cur_utt_text"])
        
        this_prompt.append(f"# Now give me the {self.phi} different questions. Don't say any other words. Don't generate sequence number of indicator. Just write questions. each question on one line.")
        
        this_prompt = "\n\n".join(this_prompt)
        
        return this_prompt
    

    def parse_returned_text(self, text):

        text = text.strip()

        try:
            splits = text.split('\n')
            result_list = []

            for i in range(len(splits)):
                if splits[i].startswith("#"):
                    splits[i] = splits[i][1:]
                if splits[i] == "\n":
                    continue
                
                result_list.append(remove_leading_number(splits[i]).strip())

            return result_list
        except Exception as e:
            print(e)
        

from utils import (
    save_turns_to_topiocqa, 
    load_turns_from_topiocqa,
)




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_query_path", type=str, default="")

    parser.add_argument("--output_query_path", type=str, default="")

    parser.add_argument("--demo_file", type=str, default="")

    parser.add_argument("--cache_dir", type=str, default="")


    parser.add_argument("--reformulation_name", type = str, default="rar", choices=[
        "vllm_mistral_GRF",
        "vllm_mistral_QR",
        ]
    ) 
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn", force=True)

    args = get_args()

    # print the arguments
    print(args)

    input_query_path = args.input_query_path
    rewrite_model = args.rewrite_model
    demo_file = args.demo_file
    reformulation_name = args.reformulation_name


    ###########################
    ## load prompter
    ###########################


    if "GRF" in reformulation_name:
        prompter = GRF_Prompter(phi=10)

    if "QR" in reformulation_name:
        if "ctx" in reformulation_name:
            prompter = QR_Prompter(phi=10, enable_context = True)
        else:
            prompter = QR_Prompter(phi=10)




    #################################
    ## load topic file and rewrite
    #################################

    turn_list = load_turns_from_topiocqa(input_query_path)

    prompts = [ 
        prompter.build_turn_prompt(turn) for turn in turn_list]
   

    outputs = generate_with_multi_gpu_vllm(
        prompts,
        model_path = args.cache_dir,
        num_gpus = 4,
        temperature= 0,
        
    )

    # dump the outputs to a file
    try:
        with open(args.output_query_path + "_temp", "w") as f:
            json.dump([output for output in outputs], f)
    except Exception as e:
        print(e)
    
    for index, output in enumerate(outputs):
            
        if "GRF" in reformulation_name:
            turn = turn_list[index]
            liste = prompter.parse_returned_text(output.outputs[0].text)

            if liste == None:
                print(f"error with turn id {turn["sample_id"]}")
                print(output.outputs[0].text)
                continue

            if len(liste) < 10:
                # append " " to the end of the list
                for i in range(10 - len(liste)):
                    liste.append("GG")

            liste = liste[:10]
            turn["GRF"] = []

            for i in range(len(liste)):
                query = liste[i]
                turn["GRF"].append(query)
                
            try:
                print("#########################")
                print("this is turn: ", turn["sample_id"])
                print(f"original query: {turn["cur_utt_text"]}")
                for i in range(len(liste)):
                    print(f"GRF_{i+1}: {liste[i]}")

            except Exception as e:
                print(f"print error with turn id {turn["sample_id"]}")
                continue


        elif "QR" in reformulation_name:

            turn = turn_list[index]
            liste = prompter.parse_returned_text(output.outputs[0].text)

            if liste == None:
                print(f"error with turn id {turn["sample_id"]}")
                print(output.outputs[0].text)
                continue

            if len(liste) < 10:
                # append " " to the end of the list
                for i in range(10 - len(liste)):
                    liste.append("GG")

            liste = liste[:10]
            turn["rewrites"] = []

            for i in range(len(liste)):
                query = liste[i]
                turn["rewrites"].append(query)
                
            try:
                print("#########################")
                print("this is turn: ", turn["sample_id"])
                print(f"original query: {turn["cur_utt_text"]}")
                for i in range(len(liste)):
                    print(f"query_{i+1}: {liste[i]}")

            except Exception as e:
                print(f"print error with turn id {turn["sample_id"]}")
                continue

    
    #################################
    ## save turn list
    #################################

    #save_turns_to_json(turn_list, args.output_query_path)
    save_turns_to_topiocqa(turn_list, args.output_query_path)
    


         
