import ollama
import re

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

    word_to_check = re.search(r'word (\w+)', prompt).group(1)
    letter_to_check = re.search(r'letter "(\w)"', prompt).group(1)

    sentences = generate_sentences(prompt)
    
    print("Generated Sentences:\n", sentences)
    
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
        if "letter" in sentence:
            verified_output = verify_output(prompt, sentence, word_to_check, letter_to_check)
            print("\nVerified Output:\n", verified_output)
        else:
            print("\nSentence:\n", sentence)

# Execute the main function
if __name__ == "__main__":
    main()
