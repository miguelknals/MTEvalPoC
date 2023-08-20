#!/bin/sh
echo 'Model paraphrase-multilingual-mpnet-base-v2'
echo 'languge SPA - ref HT'
python SBERT_bitext_score.py --src=./text/1216535_src.txt --tgt=./text/1216535_tgt.txt --outf=./text/1216535_SBERT_results.txt --ref=HT --lang=SPA --model=paraphrase-multilingual-mpnet-base-v2 

echo 'languge SPA - ref MT'
python SBERT_bitext_score.py --src=./text/1216535_src.txt --tgt=./text/1216535_tgt_MT.txt --outf=./text/1216535_SBERT_results.txt --ref=MT --lang=SPA --model=paraphrase-multilingual-mpnet-base-v2 

echo 'languge SPA - ref NEGERR'
python SBERT_bitext_score.py --src=./text/1216535_src.txt --tgt=./text/1216535_tgt_NEGATIVE.txt --outf=./text/1216535_SBERT_results.txt --ref=NEGERR --lang=SPA --model=paraphrase-multilingual-mpnet-base-v2 

echo 'languge SPA - ref NOUNERR'
python SBERT_bitext_score.py --src=./text/1216535_src.txt --tgt=./text/1216535_tgt_QRADAR.txt --outf=./text/1216535_SBERT_results.txt --ref=NOUNERR --lang=SPA --model=paraphrase-multilingual-mpnet-base-v2 

