#!/bin/sh
echo 'Model paraphrase-xlm-r-multilingual-v1'
echo 'languge SPA - ref GOLD'
python SBERT_bitext_score.py --src=./text/ra_source.txt --tgt=./text/ra_target.txt --outf=results.txt --ref=GOLD --lang=SPA --model=paraphrase-xlm-r-multilingual-v1 

echo 'languge SPA - ref WATSON'
python SBERT_bitext_score.py --src=./text/ra_source.txt --tgt=./text/ra_target_watson.txt --outf=results.txt --ref=WATSON --lang=SPA --model=paraphrase-xlm-r-multilingual-v1 

echo 'languge SPA - ref DEEPL'
python SBERT_bitext_score.py --src=./text/ra_source.txt --tgt=./text/ra_target_deepl.txt --outf=results.txt --ref=DEEPL --lang=SPA --model=paraphrase-xlm-r-multilingual-v1 

echo 'languge SPA - ref GOOGLE'
python SBERT_bitext_score.py --src=./text/ra_source.txt --tgt=./text/ra_target_google.txt --outf=results.txt --ref=GOOGLE --lang=SPA --model=paraphrase-xlm-r-multilingual-v1 

