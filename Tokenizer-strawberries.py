def char_level_tokenization(word):
    """
    Tokenizes the word at the character level.
    
    Args:
        word (str): The word to be tokenized.
        
    Returns:
        list: A list of characters.
    """
    return list(word)

def count_character(tokens, char):
    """
    Counts the occurrences of a character in the token list.
    
    Args:
        tokens (list): The list of character tokens.
        char (str): The character to be counted.
        
    Returns:
        int: The number of occurrences of the character.
    """
    return tokens.count(char)

def validate_count(word, char):
    """
    Validates the count of the character in the word.
    
    Args:
        word (str): The original word.
        char (str): The character to be counted.
        
    Returns:
        str: Validation message.
    """
    tokens = char_level_tokenization(word)
    count = count_character(tokens, char)
    actual_count = word.count(char)
    
    if count == actual_count:
        return f"The count of '{char}' in the word '{word}' is correct and is {count}."
    else:
        return f"The count of '{char}' in the word '{word}' is incorrect. Expected {actual_count}, but got {count}."

def main():
    word = input("Please enter a word: ")
    char = input("Please enter a character to count: ")
    
    if len(char) != 1:
        print("Please enter a single character.")
        return
    
    # Tokenize at character level
    tokens = char_level_tokenization(word)
    print(f"Character Tokens: {tokens}")
    
    # Count the occurrences of the character
    count = count_character(tokens, char)
    print(f"Count of '{char}': {count}")
    
    # Validate the count
    validation_message = validate_count(word, char)
    print(validation_message)

if __name__ == "__main__":
    main()
