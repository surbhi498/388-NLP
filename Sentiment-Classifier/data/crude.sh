### Crude tokenization and normalization - command line use
###  Reference: 2.4.1 Unix Tools 

## Use crude tokenization on txt
tr 'A-Za-z' '\n' <train.txt
# result: atypical reviews (standard text; reviews do not show stylistic writing e.g. "THE MOVIE WAS GOOD")

## Case normalization (Asc order) - Collapse uppercase to lowercase
tr 'A-Za-z' '\n' <train.txt | tr A-Z a-z | sort | uniq -c | sort -n -r 
# tip: verify stopword list from punkt (possibly add stopwords from top 20)
