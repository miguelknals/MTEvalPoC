# environment biclean2
#
# This program is based on https://github.com/Unbabel/COMET information
# Evaluates a MT using a reference file
# 
# COMET model suggested is hardcoded -> Default model: Unbabel/wmt22-comet-da 
# - This model uses a reference-based regression approach and is built on top 
# of XLM-R. It has been trained on direct assessments from WMT17 to WMT20 and 
# provides scores ranging from 0 to 1, where 1 represents a perfect translation.
#  (https://huggingface.co/Unbabel/wmt22-comet-da)


"""
Usage:
    COMET_bitext_and_ref_score.py --src=<file> --tgt_HT==<file> --tgt_MT=<file>  --ref=<reference> --lang=<lang>  [options]
    
Options:
    -h --help                               show this screen.
    --src=<file>                            source *_src.txt file
    --tgt_HT=<file>                         target *_tgt.txt file HUMAN Translatiom
    --tgt_MT=<file>                         target *_src.txt file MACHINE Translatiom
    --outf=<file>                           output file for results (append mode)
    --ref=<reference>                       reference label (for information reasons )
    --lang=<file>                           lang (for information reasons)
    --verbose                               prints all sentences and scores 
"""


from docopt import docopt
import math
from comet import download_model, load_from_checkpoint
import statistics





# main
if __name__ == "__main__" :
    """ Main func.
    """    
    args = docopt(__doc__)
    
    FileSource=args["--src"]    
    FileTarget_HT=args["--tgt_HT"]    
    FileTarget_MT=args["--tgt_MT"]    
    FileOutput=args["--outf"]
    reference=args["--ref"]
    language=args["--lang"]
    verbose=False
    if args['--verbose']:
        verbose=True
    if True: #verbose:
        print ("****************************************")        
        print ("COMET_bitext_and_ref_score")
        print ("****************************************")
        print ("source file      -> {}".format(FileSource))
        print ("target file (HT) -> {}".format(FileTarget_HT))
        print ("target file (MT) -> {}".format(FileTarget_MT))
        print ("output file      -> {}".format(FileOutput))
        print ("language         -> {}".format(language))
        model_name="Unbabel/wmt22-comet-da"
        print ("model (hardcoded) -> {}".format(model_name))
        
   
     
            
    nitems=0 # 0 means untill the end.
    source_sentence_list=[]
    target_MT_sentence_list=[]
    target_MT_sentence_list=[]
    target_HT_sentence_list=[]
    data=[]
    n_sentence=0
    with open(FileSource, 'r', encoding="utf-8") as file_source, \
        open(FileTarget_MT, 'r', encoding="utf-8") as file_target_MT, \
        open(FileTarget_HT, 'r', encoding="utf-8") as file_target_HT:       
        while True:
            line_source=file_source.readline()
            line_target_MT=file_target_MT.readline()
            line_target_HT=file_target_HT.readline()
            if not line_source:
                break
            n_sentence+=1            
            line_source=line_source.replace("\n","").replace("\t","")
            line_target_MT=line_target_MT.replace("\n","").replace("\t","")
            line_target_HT=line_target_HT.replace("\n","").replace("\t","")
            source_sentence_list.append(line_source)
            target_MT_sentence_list.append(line_target_MT)
            target_HT_sentence_list.append(line_target_HT)
            Dictionary_line = {}
            Dictionary_line["src"]=line_source
            Dictionary_line["mt"]=line_target_MT
            Dictionary_line["ref"]=line_target_HT
            data.append(Dictionary_line)
            nitems-=1
            if nitems==0:
                break
            
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    
    
    model_output = model.predict(data, batch_size=8)

    print(model_output)
    
    with open("V.COMETREF"+ "." + model_name.replace("/","_") + "." +language + "."+ reference + \
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
    
    

        
    
    