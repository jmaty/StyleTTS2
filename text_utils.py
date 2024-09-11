# IPA Phonemizer: https://github.com/bootphon/phonemizer

_pad = "$"
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# _special_ipa_symbols = '~'
# _special_ipa_list = ['ɔ̃', 'œ̃', 'ɑ̃']

# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)# + list(_special_ipa_symbols)

dicts = {}
for i, s in enumerate(symbols):
    dicts[s] = i

class TextCleaner:
    def __init__(self, dummy=None):
        self.word_index_dictionary = dicts
    
    def __call__(self, text):
        indexes = []
        for char in text:
            try:
                indexes.append(self.word_index_dictionary[char])
            except KeyError:
                # JMa:
                print(f'[!] Character  {char} not defined!\n    Utterance: {text}')
        return indexes
