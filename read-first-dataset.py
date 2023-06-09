# Simple Way to Read TSV Files in Python using pandas
# importing pandas library
import pandas as pd
 
# Passing the TSV file to
# read_csv() function
# with tab separator
# This function will
# read data from file

# interviews_df = pd.read_csv('corpus3/collection.tsv', sep='\t')
 
# # printing data
# print(interviews_df)

import csv

doc_set = {}
counterId = 0
with open("corpus1/collection.tsv") as tsvfile:
    tsvreader = csv.reader(tsvfile, delimiter="\t")
    for line in tsvreader:
        counterId = counterId + 1
        with open("corpus1/"+ str(counterId) +".text", 'w') as f:
            f.write(line[1])
