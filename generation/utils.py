from typing import List, Any
import json

def save_turns_to_topiocqa(
    turns: List[dict],
    output_file_path: str
) -> Any:

    '''
    save a list of dictionary to the topiocqa topic file.
    '''

    # write each dictionary in the list to a line to form a jsonl file
    with open(output_file_path, "w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n") 

def load_turns_from_topiocqa(
    topiocqa_topic_file: str
) -> List[dict]:
    '''
    Load a list of dictionary from the topiocqa topic file.
    '''

    # laod a jsonl file
    with open(topiocqa_topic_file, "r", encoding="utf-8") as f:
        data = [json.loads(line) for
                line in f.readlines()]
    
    return data