# IPA Phonemizer: https://github.com/bootphon/phonemizer

class TextCleaner:
    def __init__(self,
                 pad='$',
                 punctuation=';:,.!?¡¿—…"«»“” ',
                 letters='ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz',
                 ipa_phones="ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ",
                 ):
        # Make a unique list of symbols
        self._symbols = list(dict.fromkeys(list(pad) + list(punctuation) + list(letters) + list(ipa_phones)))

        assert len(self) == 178, f'Number of symbols must be 178 but it is {len(self)}'

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
        for i, s in enumerate(self._symbols):
            self.word_index_dict[s] = i

    def declean(self, indexes):
        return ''.join([self._symbols[i] for i in indexes])

    def __len__(self):
        return len(self._symbols)

    @property
    def symbols(self):
        return self._symbols
