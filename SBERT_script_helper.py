# environment biclean2
#

#model_l=["distiluse-base-multilingual-cased-v2",
#         "distiluse-base-multilingual-cased-v1"
#         ]


# models 

model_l=["all-MiniLM-L12-v2",
"all-MiniLM-L6-v2",
"all-distilroberta-v1",
"all-mpnet-base-v2",
"distiluse-base-multilingual-cased-v1",
"distiluse-base-multilingual-cased-v2",
"multi-qa-MiniLM-L6-cos-v1",
"multi-qa-distilbert-cos-v1",
"multi-qa-mpnet-base-dot-v1",
"paraphrase-MiniLM-L3-v2",
"paraphrase-albert-small-v2",
"paraphrase-multilingual-MiniLM-L12-v2",
"paraphrase-multilingual-mpnet-base-v2"]

model_l=["distiluse-base-multilingual-cased-v1",
         "distiluse-base-multilingual-cased-v2",
"paraphrase-multilingual-MiniLM-L12-v2",
"paraphrase-multilingual-mpnet-base-v2"]

model_l=["paraphrase-xlm-r-multilingual-v1"]





data_l=[
    ["SPA", "GOLD","./text/ra_source.txt","./text/ra_target.txt"],
    ["SPA", "WATSON","./text/ra_source.txt","./text/ra_target_watson.txt"],
    ["SPA", "DEEPL","./text/ra_source.txt","./text/ra_target_deepl.txt"],
    ["SPA", "GOOGLE","./text/ra_source.txt","./text/ra_target_google.txt"]
    ]



mask1="python SBERT_bitext_score.py --src={} --tgt={} --outf={} --ref={} --lang={} --model={} " #--verbose
with open("SBERT_script_helper.sh", 'w', encoding="utf-8") as file_helper:
    file_helper.write("#!/bin/sh\n")
    for model in model_l:
        file_helper.write("echo 'Model {}'\n".format(model))
        for item in data_l:
            src_file=item[2]
            tgt_file=item[3]
            outf="results.txt"
            ref=item[1]
            language=item[0]
            cmd1=mask1.format(src_file,tgt_file,outf,ref,language,model)
            file_helper.write("echo 'languge {} - ref {}'\n".format(language,ref))            
            file_helper.write("{}\n".format(cmd1))
            file_helper.write("\n")
            
