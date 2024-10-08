import ollama
import re
from collections import defaultdict

# Advanced Tokenizer with Enhanced Capabilities
class AdvancedTokenizer:
    def __init__(self):
        pass

    def tokenize(self, text):
        # Character-level tokenization
        return list(text)
    
    def detokenize(self, tokens):
        # Character-level detokenization
        return ''.join(tokens)

    def preprocess(self, text):
        # Custom pre-processing rules
        text = text.strip()
        return text

    def postprocess(self, text):
        # Post-processing steps
        return text

    def validate(self, text):
        # Error handling and validation
        if not text:
            raise ValueError("The input text is empty.")
        return text

# Function to directly count the occurrences of a specific letter in a word
def count_letter(word, letter):
    return word.lower().count(letter.lower())

# Function to validate if sentences end with a specific word
def validate_sentences(sentences, end_word):
    invalid_sentences = []
    for sentence in sentences:
        if not sentence.strip().endswith(end_word):
            invalid_sentences.append(sentence)
    return invalid_sentences

# Function to correct sentences that do not end with the specified word
def correct_sentences(sentences, end_word):
    corrected_sentences = []
    for sentence in sentences:
        if not sentence.strip().endswith(end_word):
            sentence = sentence.strip() + " " + end_word
        corrected_sentences.append(sentence)
    return corrected_sentences

# Function to generate sentences using Llama3 model
def generate_sentences(prompt):
    stream = ollama.chat(
        model='llama3',
        messages=[{'role': 'user', 'content': prompt}],
        stream=True,
    )
    
    response = ""
    for chunk in stream:
        response += chunk['message']['content']
    
    sentences = response.split('.')
    sentences = [sentence + '.' for sentence in sentences if sentence]
    
    return sentences

# Function to verify the AI's output for specific tasks
def verify_output(task, ai_output, word, letter):
    correct_count = count_letter(word, letter)
    ai_count_match = re.search(r'(\d+)', ai_output)
    
    if ai_count_match:
        ai_count = int(ai_count_match.group(1))
        if ai_count != correct_count:
            return f"The correct number of '{letter}' in '{word}' is {correct_count}."
        return ai_output
    else:
        return "The AI did not provide a count."

# Main function to run the script
def main():
    # User input for the prompt and the end word
    prompt = input("Enter your prompt: ")
    end_word = input("Enter the word that each sentence should end with (or press Enter to skip): ")

    # Improved regex patterns to handle variations in user input
    word_to_check_match = re.search(r'word\s+"?(\w+)"?', prompt, re.IGNORECASE)
    letter_to_check_match = re.search(r'letters?\s+"?(\w)"?', prompt, re.IGNORECASE)

    if word_to_check_match and letter_to_check_match:
        word_to_check = word_to_check_match.group(1)
        letter_to_check = letter_to_check_match.group(1)
    else:
        print("Error: Could not extract the word or letter to check from the prompt.")
        if word_to_check_match:
            print(f"Extracted word: {word_to_check_match.group(1)}")
        else:
            print("No word could be extracted.")
        if letter_to_check_match:
            print(f"Extracted letter: {letter_to_check_match.group(1)}")
        else:
            print("No letter could be extracted.")
        return

    tokenizer = AdvancedTokenizer()

    # Preprocess the prompt
    processed_prompt = tokenizer.preprocess(prompt)
    
    # Generate sentences from Llama3
    sentences = generate_sentences(processed_prompt)
    
    print("\nLlama 3 response generated by the prompt:\n", sentences)
    
    # Validate the sentences if an end word is provided
    if end_word:
        invalid_sentences = validate_sentences(sentences, end_word)
    
        if invalid_sentences:
            print("\nInvalid Sentences Found:")
            for sentence in invalid_sentences:
                print(sentence)
            
            # Correct the invalid sentences
            corrected_sentences = correct_sentences(sentences, end_word)
            
            print("\nCorrected Sentences:\n", corrected_sentences)
        else:
            print("\nAll sentences are valid.")
    
    # Verify the AI's output for specific tasks
    for sentence in sentences:
        if "letter" in sentence.lower():
            verified_output = verify_output(prompt, sentence, word_to_check, letter_to_check)
            print("\nLlama3 AI agent Critic output:\n", verified_output)
        else:
            print("\nSentence:\n", sentence)

    # Post-process the output
    final_output = tokenizer.postprocess(sentences)
    print("\nLlama3 output using the advanced tokenizer:\n", final_output)

# Execute the main function
if __name__ == "__main__":
    main()
