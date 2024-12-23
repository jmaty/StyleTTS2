import csv
import re


class TextCleaner:
    """
    Class for handling with phonemes as tokens
    """

    def __init__(self, symbols, pad="_"):
        """Init

        Args:
            symbols (dict, string): Phoneme-token encoding dict
            pad (str, optional): Symbol for padding. Defaults to '_'.
        """
        # Load symbol encodin dict
        self._symbols = symbols if isinstance(symbols, dict) else load_symbol_dict(symbols)
        self._pad = pad
        # assert len(self) == 81, f'Number of symbols must be 81 but it is {len(self)}'
        assert pad in self._symbols, f"Pad symbol ({pad}) is not included in symbols!"

    def __call__(self, text):
        indexes = []
        for c in text:
            try:
                indexes.append(self._symbols[c])
            except KeyError:
                # JMa:
                print(f'[!] Character  "{c}" not defined!\n    Utterance: {text}')
        return indexes

    def declean(self, indexes):
        return "".join([self._symbols[i] for i in indexes])

    def check(self, symbols):
        """
        Checks if every input symbol exists is defined.

        Args:
            symbols (str): The input string of characters to be checked.

        Returns:
            bool: True if all input symbols are defined, otherwise False.
        """
        # Convert the input string into a set of unique characters
        unique_chars = set(symbols)
        # Get the set of keys from the dictionary
        valid_symbols = set(self._symbols.keys())
        # Check if all unique characters are a subset of the dictionary keys
        return unique_chars.issubset(valid_symbols)

    def __len__(self):
        return len(self._symbols)

    def __contains__(self, symbols):
        return self.check(symbols)

    @property
    def symbols(self):
        return self._symbols

    @property
    def pad(self):
        return self._pad, self._symbols[self._pad]

    @property
    def blank(self):
        return " ", self._symbols[" "]


def load_symbol_dict(fpath):
    """Load symbol dict from a text file

    Args:
        fpath (str): path to text file with symbol definitions

    Returns:
        dict: symbol dict
    """
    with open(fpath, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=",", quotechar='"')
        symbol_dict = {row[0]: int(row[1]) for row in reader}
    return symbol_dict


def add_spaces_around_punctuation(text):
    """Add spaces around punctuation in a phonetic string.

    Args:
        text (str): phonetic string

    Returns:
        str: phonetic string with non-initial and non-final punctution surrounded by spaces
    """
    # Add a space before punctuation if it is not already preceded by a space
    text = re.sub(r"(?<! )([.,!?;:])", r" \1", text)
    # Add a space after punctuation if it is not already followed by a space
    text = re.sub(r"([.,!?;:])(?! )", r"\1 ", text)
    return text.strip()


def remove_spaces(text):
    """Remove spaces from a phonetic string.

    Args:
        text (str): phonetic string

    Returns:
        str: phonetic string with spaces removed
    """
    return text.replace(" ", "")
