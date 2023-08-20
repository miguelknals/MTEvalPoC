# this code calculates the BLEU score
# BLEU needs detokenize, we can use detokenize function or 
# use tokenize option in bleu function parameters
import os
from sacrebleu.metrics import BLEU, CHRF, TER
from sacremoses import MosesDetokenizer, MosesTokenizer
sacreM_Tok = MosesTokenizer(lang="en")




if __name__ == '__main__':     
    hyp_file = os.path.join(".","text","1216535_tgt.txt") # laika
    ref_file = os.path.join(".","text","1216535_tgt_MT.txt") # laika
    preds=[]; refs=[]  # REFS -> POST and  PRED -> PRE   
    with open (hyp_file,"r", encoding="UTF-8") as hyp, \
         open (ref_file, "r", encoding="UTF-8") as ref:
            while True: 
                l_hyp=hyp.readline().strip('\n').strip() # we need to remove ending nl and spaces
                l_ref=ref.readline().strip('\n').strip() # we need to remove ending nl and spaces
                # can detokenize here or directly in the blue stence
                l_hyp=sacreM_Tok.tokenize(l_hyp,return_str=True)
                l_ref=sacreM_Tok.tokenize(l_ref,return_str=True)
                if not l_hyp:
                    break
                preds.append(l_hyp)
                refs.append(l_ref)
                
    
    #bleu= BLEU()   
    #bleu = BLEU(trg_lang="zh",tokenize="zh")
    #bleu = BLEU(trg_lang="ja",tokenize="ja-mecab")
    bleu = BLEU()
    
    # Calculate and print the BLEU score
    #refs=refs # Yes, it is a list of list(s) as required by sacreBLEU
    # PREDS <> PRETRANS <> MACHINE    REFS <> POSTTRANS <> HUMAN
    res =bleu.corpus_score(preds, [refs])
    print (res)
    print (bleu.get_signature())
    
    chrf = CHRF()
    reschrf=chrf.corpus_score(preds, [refs])
    print (reschrf)
    
    ter=TER()
    resTER=ter.corpus_score(preds, [refs])
    print (resTER)
    
    # meteor
        
    print ("EOP.")