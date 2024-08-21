# Data Preparation

Follow the instructions below to prepare data of each language. 

## MERLIN (German, Italian, Czech)
1. Download **TXT-files** from https://www.merlin-platform.eu/.
2. Extract all txt files from the `meta_ltext` folder of the downloaded `merlin-text-v1.1.zip`.
3. Place all German txt files into `./data/ori/de/merlin`; all Italian txt files into `./data/ori/it/merlin`, and all Czech txt files into `./data/ori/cs/merlin`.

## Write \& Improve (English)
1. Download **W&I+LOCNESS v2.1** from https://www.cl.cam.ac.uk/research/nl/bea2019st/.
2. Extract `{A|B|C}.{train|dev}.json` from the `json` folder of the downloaded `wi+locness_v2.1.bea19.tar.gz` into `./data/ori/en/write_and_improve`.

## CEDEL2 (Spanish)
1. In the webpage http://cedel2.learnercorpora.com/search_simple, click "Search" then "Download" with choosing the **Texts with metadata** download format.
2. Extract all txt files from the downloaded `texts.zip` into `./data/ori/es/cedel2`.

## COPLE2 (Portuguese)
1. In the webpage http://teitok.clul.ul.pt/cople2/index.php?action=cqp, click "Search", navigate into the webpage of each xml (by clicking each xml file in the "ID" field) and click "Download text".
2. Additionally, save the meta data (ID, Months of PT and so on) of each xml.
3. Merge the meta data and the texts into a tsv file named `cople2.tsv` with fields: ID, Months of PT, Proficiency, Nationality, Mother tongue, Foreign languages, Genre, Prompt, Topic, Number of words, Type of text, text. The `text` field stores the downloaded texts and the other fields stores the saved metadata. Place the merged tsv file `cople2.tsv` into `./data/ori/pt/cople2`.

## Structure of `./data/ori`

The structure of `./data/ori` outputted by `ls -R ./data/ori` is as follows. Use the output to check if your data preparation process is correct. 

```bash
./data/ori:
cs
de
en
es
it
pt

./data/ori/cs:
merlin

./data/ori/cs/merlin:
0601.txt
(more files omitted)

./data/ori/de:
merlin

./data/ori/de/merlin:
1023_0001416.txt
(more files omitted)

./data/ori/en:
write_and_improve

./data/ori/en/write_and_improve:
A.dev.json
A.train.json
B.dev.json
B.train.json
C.dev.json
C.train.json

./data/ori/es:
cedel2

./data/ori/es/cedel2:
ar_wr_12_26_0.5_14_eh.txt
(more files omitted)

./data/ori/it:
merlin

./data/ori/it/merlin:
1325_1001008.txt
(more files omitted)

./data/ori/pt:
cople2

./data/ori/pt/cople2:
cople2.tsv
```