cat $1 | while IFS= read -r line; do
  # Split the line at the first '|'
  before_pipe=$(echo "$line" | cut -d'|' -f1)
  after_pipe=$(echo "$line" | cut -d'|' -f2-)
  
  # Only process the part after the pipe
  processed=$(echo "$after_pipe" | 
    sed 's/["'\''"„«»]//g' |
    sed 's/[,;:…()]/#/g' | # Nahrazení středníků (;) a dvojteček (:) čárkou (,)
    sed 's/ \(-\) / , /g; s/^\(-\) /, /; s/ \(-\)$/ # /' | # Nahrazení pomlčky čárkou, pokud je alespoň z jedné strany mezera
    sed 's/\([^0-9]\)-\([^0-9]\)/\1\2/g; s/^-//; s/-$//' | # Odstranění pomlčky, pokud nesousedí s číslovkou
    sed 's/[$%Ç*]/#/g' | # Náhrada "pauzových" symbolů znakem #
    sed 's/ *#\(\s*#\)* */ # /g' | # Spojení více # (i když jsou oddělena mezerami) do jednoho
    sed 's/| *# */|/g; s/[[:space:]]*#$//' | # Nahraď |# za | a smaž # na konci řádky
    sed 's/\s*#/,/g' | # Nahradí # čárkou
    sed 's/^,//' | # Odstraní interpunkci na začátku řádky
    sed 's/[[:space:]]\{2,\}/ /g' | # Nahraď vícenásobné mezery jednou mezerou
    sed 's/ \+,/,/g' | # Zajisti, ať před čárkou není mezera
    sed -E 's/([[:punct:]])[[:punct:]]?,/\1/g' | # Nahraď ., apod za .
    sed 's/^[[:space:]]*//' | # Smaž všechny bíle znaky na začátku řádky
    sed 's/[[:space:]]*$//' | # Smaž všechny bíle znaky na konci řádky
    sed '/[[:punct:]]$/!s/$/,/' | # Pokud řádek nekončí interpunkcí, přidej čárku
    sed 's/[[:space:]]\{2,\}/ /g') # Nahraď vícenásobné mezery jednou mezerou
  
  # Output the result
  echo "${before_pipe}|${processed}"
done
