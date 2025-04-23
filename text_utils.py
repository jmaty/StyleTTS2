import csv
import re
from logger import get_logger

# Setup logger
logger = get_logger(__name__)


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
        # Load symbol encoding dict
        self._symbols = symbols if isinstance(symbols, dict) else load_symbol_dict(symbols)
        self._pad = pad
        # assert len(self) == 81, f'Number of symbols must be 81 but it is {len(self)}'
        assert pad in self._symbols, f"Pad symbol ({pad}) is not included in symbols!"

        # Pre-compile regex for adding spaces
        # Add a space before punctuation if it is not already preceded by a space
        self._re_before_punctuation = re.compile(r"(?<! )([.,!?;:])")
        # Add a space after punctuation if it is not already followed by a space
        self._re_after_punctuation = re.compile(r"([.,!?;:])(?! )")

    def __call__(self, text, pad=False):
        """Call method for converting a phonetic string into a list of token IDs.

        Args:
            text (str): phonetic string

        Returns:
            list: list of token IDs
        """
        # logger.debug("Cleaning text: %s", text)

        indexes = [self.pad[1]] if pad else []
        # Add spaces around punctuation and convert the text into a list of token IDs
        for c in self.add_spaces_around_punctuation(text):
            try:
                indexes.append(self._symbols[c])
            except KeyError:
                logger.warning("[!] Character '%s' not defined in '%s'!", c, text)
        if pad:
            indexes.append(self.pad[1])

        # logger.debug("Token IDs: %s", indexes)
        return indexes

    def declean(self, indexes):
        """Convert a list of token IDs into a phonetic string.

        Args:
            indexes (list): list of token IDs

        Returns:
            str: phonetic string
        """
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
        """Return the number of symbols in the symbol dict.

        Returns:
            int: number of symbols
        """
        return len(self._symbols)

    def __contains__(self, symbols):
        """Check if all input symbols are defined.

        Args:
            symbols (str): The input string of characters to be checked.

        Returns:
            bool: True if all input symbols are defined, otherwise False.
        """
        return self.check(symbols)

    @property
    def symbols(self):
        """Return the symbol dict.

        Returns:
            dict: symbol dict
        """
        return self._symbols

    @property
    def pad(self):
        """Return the pad symbol and its corresponding token ID.

        Returns:
            tuple(str, int): pad symbol and its corresponding token ID
        """
        return self._pad, self._symbols[self._pad]

    @property
    def blank(self):
        """Return the blank symbol and its corresponding token ID.

        Returns:
            tuple(str, int): blank symbol and its corresponding token ID
        """
        return " ", self._symbols[" "]

    def add_spaces_around_punctuation(self, text):
        """Add spaces around punctuation in a phonetic string.

        Args:
            text (str): phonetic string

        Returns:
            str: phonetic string with non-initial and non-final punctution surrounded by spaces
        """
        # Add a space before punctuation if it is not already preceded by a space
        text = self._re_before_punctuation.sub(r" \1", text)
        # Add a space after punctuation if it is not already followed by a space
        text = self._re_after_punctuation.sub(r"\1 ", text).strip()

        # logger.debug("Text after adding spaces: %s", text)
        return text

    @staticmethod
    def remove_spaces(text):
        """Remove spaces from a phonetic string.

        Args:
            text (str): phonetic string

        Returns:
            str: phonetic string with spaces removed
        """
        return text.replace(" ", "")


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
