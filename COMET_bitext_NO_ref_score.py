# environment biclean2
#
# This program is based on https://github.com/Unbabel/COMET information
# Evaluates a MT WITHOUT usign a reference file
# 
# COMET model suggested is hardcoded -> Reference-free: Unbabel/wmt22-cometkiwi-da
# - This reference-free model uses a regression approach and is built on top of
# InfoXLM. It has been trained on direct assessments from WMT17 to WMT20, as
# well as direct assessments from the MLQE-PE corpus. Like the default model,
# it also provides scores ranging from 0 to 1.
# https://huggingface.co/Unbabel/wmt22-cometkiwi-da
# typical usage command line:
# 


"""
Usage:
    COMET_bitext_NO_ref_score.py --src=<file> --tgt_MT=<file> \
        --token=<token>  --outf=<file> --lang=<lang> --ref=<ref>  [options]
    
Options:
    -h --help                               show this screen.
    --src=<file>                            source *_src.txt file
    --tgt_MT=<file>                         target *_tgt.txt file MACHINE Translatiom
    --outf=<file>                           output file for results (append mode)
    --toke=<token>                           token to access COMET library
    --ref=<reference>                       reference label (for information reasons )
    --lang=<file>                           lang (for information reasons)
    --verbose                               prints all sentences and scores 
"""


from docopt import docopt
import math
from comet import download_model, load_from_checkpoint
from huggingface_hub import login
import os

import statistics

#  (https://huggingface.co/Unbabel/wmt22-comet-da)

MIN_WORDS_TO_STUDY=5
MAX_WORDS_TO_STUDY=50
# main
if __name__ == "__main__" :
    """ Main func.
    """    
    args = docopt(__doc__)
    
    FileSource=args["--src"]    
    FileTarget_MT=args["--tgt_MT"]    
    FileOutput=args["--outf"]
    Token=args["--token"]
    reference=args["--ref"]
    language=args["--lang"]
    verbose=False
    if args['--verbose']:
        verbose=True
    if True: #verbose:
        print ("****************************************")        
        print ("COMET_bitext_NO_ref_score")
        print ("****************************************")
        print ("source file      -> {}".format(FileSource))
        print ("target file (MT) -> {}".format(FileTarget_MT))
        print ("output file      -> {}".format(FileOutput))
        print ("language         -> {}".format(language))
        model_name="Unbabel/wmt22-cometkiwi-da"
        print ("model (hardcoded) -> {}".format(model_name))
        print ("Sentences between {} and {} words".format(MIN_WORDS_TO_STUDY, MAX_WORDS_TO_STUDY))
        # try to get path of results output file
        #basenamepath=FileOutput.replace(os.path.basename(FileOutput), "")
        
   
     
            
    nitems=-1300 # 0 means untill the end.
    source_sentence_list=[]
    target_MT_sentence_list=[]
    target_MT_sentence_list=[]
    data=[]
    n_sentence=0
    with open(FileSource, 'r', encoding="utf-8") as file_source, \
        open(FileTarget_MT, 'r', encoding="utf-8") as file_target_MT:
        while True:
            line_source=file_source.readline()
            line_target_MT=file_target_MT.readline()
            if not line_source:
                break
            num_words= len(line_source.split())
            if num_words >= MIN_WORDS_TO_STUDY and num_words <= MAX_WORDS_TO_STUDY:
                n_sentence+=1            
                line_source=line_source.replace("\n","").replace("\t","")
                line_target_MT=line_target_MT.replace("\n","").replace("\t","")
                source_sentence_list.append(line_source)
                target_MT_sentence_list.append(line_target_MT)
                Dictionary_line = {}
                Dictionary_line["src"]=line_source
                Dictionary_line["mt"]=line_target_MT
                data.append(Dictionary_line)
                nitems-=1
            if nitems==0:
                break
            
    login(token=Token)
    
    model_path = download_model(model_name)
    
    model = load_from_checkpoint(model_path)
    
    
    model_output = model.predict(data, batch_size=8,gpus=2)

    print(model_output)
    
    with open(FileOutput+ "." + "V.COMETNOREF"+ "." + model_name.replace("/","_") + "." +language + "."+ reference + \
           ".csv", 'w', encoding="utf-8") as file_values:
        for score, src_stc, tgt_MT_stc in zip (model_output[0], \
                source_sentence_list, target_MT_sentence_list):
            src_stc=src_stc.replace("\t","") # remove tab
            tgt_MT_stc=tgt_MT_stc.replace("\t","") # remove tab            
            file_values.write("{}\t{}\t{}\n".format(score, src_stc, tgt_MT_stc))
            n_sentence+=1
            if verbose:
                print ("---------------")
                print (src_stc)
                print (tgt_MT_stc)
                print (score)
    
    
    mean=statistics.mean(model_output[0])
    std_dev=statistics.stdev(model_output[0])
    #print (data)
    if True: #verbose:    
        print ("Sentences   -> {}".format(n_sentence))
        print ("Mean        -> {}".format(mean))
        print ("Std.dev     -> {}".format(std_dev))
    
    
    with open(FileOutput, 'a', encoding="utf-8") as file_output:
        file_output.write ("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format (FileSource,FileTarget_MT,                                                              
                reference, language, model_name,mean, std_dev))
    
    
    #with open(FileOutput, 'a', encoding="utf-8") as file_output:
    #    file_output.write ("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format (FileSource,FileTarget,                                                              
    #            reference, language, model_name,mean, std_dev))
    
    

        
    
    