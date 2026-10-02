import json
import textwrap

def flatten_and_chunk_dataset(input_filepath: str, output_filepath: str, max_width: int = 40) -> None:
    """
    Parses a nested JSON QA dataset, extracts items, and writes them
    to a flat file while bounding lines to a strict character limit.
    """
    with open(input_filepath, 'r', encoding='utf-8') as file_in:
        payload = json.load(file_in)

    with open(output_filepath, 'w', encoding='utf-8') as file_out:
        for article in payload.get('data', []):
            for paragraph in article.get('paragraphs', []):
                for qa_item in paragraph.get('qas', []):
                    question = qa_item.get('question', '').strip()
                    answers = qa_item.get('answers', [])
                    
                    if answers:
                        answer_text = answers[0].get('text', '').strip()
                        
                        # Chunking the question on word boundaries
                        for chunk in textwrap.wrap(question, width=max_width):
                            file_out.write(f"{chunk}\n")
                        
                        # Chunking the answer on word boundaries
                        for chunk in textwrap.wrap(answer_text, width=max_width):
                            file_out.write(f"{chunk}\n")

if __name__ == "__main__":
    flatten_and_chunk_dataset("train-v2.0.json", "flatten.txt")
