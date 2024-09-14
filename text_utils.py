# IPA Phonemizer: https://github.com/bootphon/phonemizer
import unicodedata

# _pad = "$"
# _punctuation = ';:,.!?¡¿—…"«»“” '
# _letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# # _special_ipa_symbols = '~'
# # _special_ipa_list = ['ɔ̃', 'œ̃', 'ɑ̃']

# # Export all symbols:
# symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)# + list(_special_ipa_symbols)

# dicts = {}
# for i, s in enumerate(symbols):
#     dicts[s] = i

class TextCleaner:
    def __init__(self,
                 pad='$',
                 punctuation=None,
                 letters=None,
                 ipa_phones=None, 
                 ):
        if punctuation is None:
            punctuation = []
        if letters is None:
            letters = []
        if ipa_phones is None:
            ipa_phones = []
        self.symbols = list(pad) + punctuation + letters + ipa_phones
        self._make_word_index_dict()
    
    def __call__(self, text):
        chars = []
        for char in text:
            if unicodedata.combining(char):
                chars[-1] += char
            else:
                chars.append(char)
        
        indexes = []
        for char in chars:
            try:
                indexes.append(self.word_index_dict[char])
            except KeyError:
                # JMa:
                print(f'[!] Character  {char} not defined!\n    Utterance: {text}')
        return indexes
    
    def _make_word_index_dict(self):
        self.word_index_dict = {}
        for i, s in enumerate(self.symbols):
            self.word_index_dict[s] = i        
