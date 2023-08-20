# reads a tab csv 3 columns 
# score, source and target
# histogram code based on 
# https://www.geeksforgeeks.org/how-to-plot-normal-distribution-over-histogram-in-python/



"""
Usage:
    file_score_report.py --csv=<file>  [options]
    
Options:
    -h --help                               show this screen.
    --csv=<file>                            csv with score file
    --verbose                               prints all sentences and scores 
"""


from docopt import docopt
import csv
import statistics
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import os

def  Read_CSV_data(csv_data_filename):
    list_score=[]
    list_src=[]
    list_tgt=[]

    with open(csv_data_filename,newline='',encoding="UTF-8") as O_csv_data_file:
            reader=csv.reader (O_csv_data_file, delimiter='\t')
                                
            for row in reader:
                list_score.append(float(row[0]))
                list_src.append(row[1])
                list_tgt.append(row[2])
    
    return list_score,list_src, list_tgt

def Mean_and_std_dev(list_score, out_file_csv):
    mean=statistics.mean(list_score)
    std_dev=statistics.stdev(list_score)
    
    # Fit a normal distribution to
    # the data:
    # mean and standard deviation
    mu, std = norm.fit(list_score) 
    
    plt.hist(list_score, bins=10, density=True, alpha=0.6, color='b')
    # Plot the PDF.
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'k', linewidth=2)
    title= "Score values histogram + normal distribution\n"
    title +=  "Mean: {:.4f} - Std.dev: {:.4f}".format(mu, std)
    plt.title(title)    
    plt.savefig(out_file_csv+".png")
    
    return
    

def Generate_report_body(list_score,list_src, list_tgt,FileCSV):
    # schema
    schema=""
    schema+="<!DOCTYPE html>\n"
    schema+="<html lang=""en"">\n"
    schema+="<head>\n"
    schema+="<meta charset=""utf-8"">\n"
    schema+="<title>{0}</title>\n"
    schema+="<!-- <link rel=""stylesheet"" href=""style.css""> -->\n"
    schema+="<!-- <script src=""script.js""></script>  -->\n"
    schema+="</head>\n"
    schema+="<body>\n"
    schema+="<!-- page content -->\n"
    # Title
    schema+="<h1>{0}</h1>\n"
    # graf info
    schema+="<h2>Comments</h2>\n"
    schema+="<p>Text</p><ul><li>Item 1</li><li>Item 2</li></ul>\n"
    
    
    schema+="<h2>{1}</h2>\n"
    schema+="<p><figure><img src=""{2}"" ></figure>\n"
    # graf best values
    schema+="<h2>{3}</h2>\n"
    schema+="<p>{4}</p>\n"
    # graf best values
    schema+="<h2>{5}</h2>\n"
    schema+="<p>{6}</p>\n"
    #
    
    schema+="</body>\n"
    schema+="</html>\n"
    
    HTML_title="{0} - Score analysis".format(FileCSV)
    HTML_H1="Score histogram and normal distribution"
    HTML_FIG1=os.path.basename(FileCSV)+".png" 
    
    # table
    HTML_pretable="<table><tr><th>src</th><th>tgt</th><th>Score</th></th><th>Comment</th></tr>\n"
    HTML_row="<tr><td>{0}</td><td>{1}</td><td>{2:.4f}</td><td>{3}</td>\n"
    HTML_posttable="</table>\n"
    
    # lets sort by 
    zipped = zip(list_score,list_src, list_tgt)
    zipped= sorted(zipped, key=lambda x: x[0])
    
    # 15 best values
    HTML_H2="15 best scored translations"    
    print (HTML_H2)
    best15=zipped[-15:] 
    HTML_TABLE2=HTML_pretable
    for best in best15:
        print ("---------------------------------")        
        print ("{0}\n{1}\n{2}\n".format(best[0],best[1],best[2]))
        HTML_TABLE2+=HTML_row.format(best[1],best[2],best[0],"")
    HTML_TABLE2+=HTML_posttable

    HTML_H3="15 worst scored translations"    
    print (HTML_H3)    
    # 15 worst values:
    worst15=zipped[0:14]
    HTML_TABLE3=HTML_pretable
    for worst in worst15:
        print ("---------------------------------")
        print ("{0}\n{1}\n{2}\n".format(worst[0],worst[1],worst[2]))
        HTML_TABLE3+=HTML_row.format(worst[1],worst[2],worst[0],"")
    HTML_TABLE3+=HTML_posttable
    
    # generate html
    html_code=schema.format(HTML_title, HTML_H1, HTML_FIG1,
                HTML_H2, HTML_TABLE2, 
                HTML_H3, HTML_TABLE3)
    
    outputHTML=FileCSV+".html"
    with open(outputHTML,"w") as f:
        f.write(html_code)
    
    return
    
        
    
        
        
    
    
                
            




if __name__ == "__main__" :
    """ Main func.
    """    
    args = docopt(__doc__)
    
    FileCSV=args["--csv"]    
    verbose=False
    if args['--verbose']:
        verbose=True
    if True: #verbose:
        print ("****************************************")        
        print ("file_score_report")
        print ("****************************************")
        print ("CSV report       -> {}".format(FileCSV))
        print ("****************************************")
        
    
    
    list_score,list_src, list_tgt=Read_CSV_data(FileCSV) 
    
    Mean_and_std_dev(list_score,FileCSV)
    
    Generate_report_body(list_score,list_src, list_tgt,FileCSV)
    
    
    
    
    
        
        