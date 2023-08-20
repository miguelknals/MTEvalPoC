# environment biclean2
#
"""
Usage:
    SBERT_bitext_score.py --src=<file> --tgt=<file> --outf=<file> --ref=<reference> --lang=<lang> --model=<model> [options]
    
Options:
    -h --help                               show this screen.
    --src=<file>                            source *_src.txt file
    --tgt=<file>                            target *_tgt.txt file
    --outf=<file>                           output file for results (append mode)
    --ref=<reference>                       reference (for information reasons)
    --lang=<file>                           lang (for information reasons)
    --model=<model>                         pretrained SBERT model (i.e. distiluse-base-multilingual-cased-v1)
    --verbose                               prints all sentences and scores 
"""

# Example python SBERT_bitex t_score.py --src=./text/ra_source.txt 
# --tgt=./text/ra_target_google.txt --outf=results.txt --ref=GOOGLE 
# --lang=SPA --model=paraphrase-xlm-r-multilingual-v1 
#
# multilingual models (0.91 0.92 0.93) 
# model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
# model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
# model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# 
# Chat GPT comment
# Keep in mind that while this approach provides similarity scores, 
# it's not exactly the same as BERTScore. BERTScore is a specialized
# metric that has been designed and evaluated to provide better 
# alignment with human judgments of translation quality. 
# If you're looking for a comprehensive and accurate evaluation, 
# BERTScore (or similar metrics) is recommended.
#
# Have not investigate if there is a BERTScore multilingual for MT


from docopt import docopt
import math
from sentence_transformers import SentenceTransformer, util 
import statistics

import torch

MIN_WORDS_TO_STUDY=5
MAX_WORDS_TO_STUDY=50


# main
if __name__ == "__main__" :
    """ Main func.
    """    
    args = docopt(__doc__)
   
    model_name=args["--model"]    
    FileSource=args["--src"]    
    FileTarget=args["--tgt"]    
    FileOutput=args["--outf"]
    reference=args["--ref"]
    language=args["--lang"]
    verbose=False
    if args['--verbose']:
        verbose=True
        
    # hacked to be verbose always
    verbose=True
    if verbose:
        print ("****************************************")        
        print ("SBERT_bitext_score")
        print ("****************************************")
        print ("source file -> {}".format(FileSource))
        print ("target file -> {}".format(FileTarget))
        print ("output file -> {}".format(FileOutput))
        print ("language    -> {}".format(language))
        print ("model       -> {}".format(model_name))
        print ("Sentences between {} and {} words".format(MIN_WORDS_TO_STUDY, MAX_WORDS_TO_STUDY))
        print ("****************************************")
        
        
            
    model = SentenceTransformer(model_name ,device='cuda:1')
    nitems=-200 # -1 means untill the end.
    source_sentence_list=[]
    target_sentence_list=[]
    with open(FileSource, 'r', encoding="utf-8") as file_source, \
        open(FileTarget, 'r', encoding="utf-8") as file_target:       
        while True:
            line_source=file_source.readline()
            line_target=file_target.readline()
            if not line_source:
                break
            num_words= len(line_source.split())
            if num_words >= MIN_WORDS_TO_STUDY and num_words  <= MAX_WORDS_TO_STUDY:
                source_sentence_list.append(line_source.replace("\n",""))
                target_sentence_list.append(line_target.replace("\n",""))
                nitems-=1
                
            if nitems==0:
                break
    
    
    source_sentence_list_embeddings = model.encode(source_sentence_list)
    target_sentence_list_embeddings = model.encode(target_sentence_list)
    
    #Print the embeddings
    #for sentence, embedding in zip(source_sentence_list, 
    #                               source_sentence_list_embeddings):
    #    print("Sentence:", sentence)
    #    print("Embedding:", embedding)
    #    print("")

    cos_sim_list=[]
    n_sentence=0
    with open(FileOutput+"."+"V.SBERT."+"."+model_name+"."+language + "."+ reference + ".csv", 'w', encoding="utf-8") as file_values:
        for src_stc, src_emb, tgt_stc, tgt_emb in zip (source_sentence_list, 
                source_sentence_list_embeddings, target_sentence_list,
                target_sentence_list_embeddings):
            cos_sim=util.cos_sim(src_emb, tgt_emb) # a tensor of tensors
            src_stc=src_stc.replace("\t","") # remove tab
            tgt_stc=tgt_stc.replace("\t","") # remove tab            
            v=cos_sim.item()
            cos_sim_list.append(v) # .item removes single element.
            file_values.write("{}\t{}\t{}\n".format(v, src_stc, tgt_stc))
            n_sentence+=1
            if verbose:
                print ("---------------")
                print (src_stc)
                print (tgt_stc)
                print (cos_sim)
    #
    mean=statistics.mean(cos_sim_list)
    std_dev=statistics.stdev(cos_sim_list)
    #
    zipped= list(zip(cos_sim_list,source_sentence_list,target_sentence_list))
    zipped.sort()
    
    
    if verbose: 
        print ("Sentenc     -> {}".format(n_sentence))
        print ("Mean        -> {}".format(mean))
        print ("Std.dev     -> {}".format(std_dev))
    
    with open(FileOutput, 'a', encoding="utf-8") as file_output:
        file_output.write ("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format (FileSource,FileTarget,                                                              
                reference, language, model_name,mean, std_dev))
 
        
    
    