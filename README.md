# 1 MTEvalPoC - Machine Translation evaluation examples (BLEU,  SBERT, COMET...)

## 1.2 Intro
We want to evaluate the quality of 3 machine translations (MT) systems: Google Translator MT,
DeepL MT and Watson MT, based on human translation (HT), but also without human translation. 
This evaluation is based on aprox 8500 words extracted from an IBM online help, that has been
 human translated (8 years ago based on the old  IBM MT model (probably a neuronal RNN)). So,
 be aware, human translations is highly biased onto Watson MT.) .There will be 3 MT translations 
 for Watson, DeepL and Google.  The exercise here to explain how to reach "actual" results.  
 
 
 ## 1.3 Files 
 
 - ra_source.txt (Source file) : (46 lines aprox, 8500 words (sentences with medium length and NO tags (originally dita files))
- ra_target.txt (Target translation in Spanish) (Human translation based on IBM's MT available at the time. 8 years ago)
- ra_target_google.txt (MT for Google Translator)  (Translated from the public Google Translator )
- ra_target_deepl.txt (MT for DeepL) (translated from the public DeepL)
-  ra_target_watson.txt (Translated with latest IBM MT)

## 1.4 1st test -  MT evaluation vs human translation (HT) (BLEU SCORE)
We will use the BLEU score, one of the most widely used.  For reference:
100 - MT matches reference 
90 -  MT is close to human translations
60 - MT can be used by a translator 
Results of HT(human translation) vs MT systems:


| Model | BLEU|
| :---:   | :---: |
| Google  | 58.07 |
| DeepL  | 57.84 |
| Watson | 68.88 |
![info](docs/Bleu_score.png)