# environment biclean2
#
"""
Usage:
    COMET_bitext_and_ref_score.py --src=<file> --tgt_HT==<file> --tgt_MT=<file>  --ref=<reference> --lang=<lang>  [options]
    
Options:
    -h --help                               show this screen.
    --src=<file>                            source *_src.txt file
    --tgt_HT=<file>                         source *_src.txt file HUMAN Translatiom
    --tgt_MT=<file>                         source *_src.txt file MACHINE Translatiom
    --tgt=<file>                            source *_src.txt file
    --outf=<file>                           output file for results (append mode)
    --ref=<reference>                       reference (for information reasons)
    --lang=<file>                           lang (for information reasons)
    --model=<model>                         pretrained SBERT model (i.e. distiluse-base-multilingual-cased-v1)
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
    
    # len (sys.argv) <3:
    #    print ("Usage: xml-sax-parser-4-xlif.py <document> ")
    #    print ("")
    #    print (" <document> Document name to parse")
    #    quit()     
    #model = SentenceTransformer('all-MiniLM-L6-v2')
    #model = SentenceTransformer('distiluse-base-multilingual-cased-v2')
    #model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    #FileSource=sys.argv[1]
    #FileTarget=sys.argv[2]
   
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
        print ("source file      -> {}".format(FileSource))
        print ("target file (HT) -> {}".format(FileTarget_HT))
        print ("target file (MT) -> {}".format(FileTarget_MT))
        print ("output file      -> {}".format(FileOutput))
        print ("language         -> {}".format(language))
        
   
     
            
    nitems=0 # 0 means untill the end.
    source_sentence_list=[]
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
            n_sentence+=1
            if not line_source:
                break
            Dictionary_line = {}
            Dictionary_line["src"]=line_source.replace("\n","")
            Dictionary_line["mt"]=line_target_MT.replace("\n","")
            Dictionary_line["ref"]=line_target_HT.replace("\n","")
            data.append(Dictionary_line)
            nitems-=1
            if nitems==0:
                break
            
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)
    
    model_output = model.predict(data, batch_size=8, gpus=2)

    print(model_output)
    mean=statistics.mean(model_output[0])
    std_dev=statistics.stdev(model_output[0])
    #print (data)
    if True: #verbose:    
        print ("Sentences   -> {}".format(n_sentence))
        print ("Mean        -> {}".format(mean))
        print ("Std.dev     -> {}".format(std_dev))
    
    
    
    
    
    #with open(FileOutput, 'a', encoding="utf-8") as file_output:
    #    file_output.write ("{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format (FileSource,FileTarget,                                                              
    #            reference, language, model_name,mean, std_dev))
    
    

        
    
    