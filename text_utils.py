# IPA Phonemizer: https://github.com/bootphon/phonemizer

class TextCleaner:
    def __init__(self,
                 pad='$',
                 punctuation=';:,.!?¡¿—…"«»“” ',
                 letters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                 ipa_phones="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
                 ):
        self.symbols = list(dict.fromkeys(list(pad) + list(punctuation) + list(letters) + list(ipa_phones)))
        self._make_word_index_dict()

    def __call__(self, text):
        indexes = []
        for char in text:
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
    
    def __len__(self):
        return len(self.symbols)
