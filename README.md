# DiffTime

Source code for our NAACL paper "Deep Neural Models of Semantic Shift".


##Installation

###Additional files

This section described the additional files that are needed to run this program.


####COHA sample files

The results generated below come from the Corpus of Historical American English (COHA) sample corpus. The sample corpus can be downloaded from [this website](https://www.corpusdata.org/formats.asp). Choose the COHA linear text samples. The unzipped files should be placed here: datasets/coha_sample/

I have not tested this, but I assume this code works equally well on the full corpus if one has access to it.

####MEN task files

The synchronic evaluation requires data from the MEN dataset (Bruni et al. 2014). One location for this data is [here](https://github.com/mfaruqui/word-vector-demo/blob/master/data/EN-MEN-TR-3k.txt). This file should be placed here: datasets/synchronic_task/

##Training

Run the example code using the following command:

    python3 fluidtime.py <arguments>

##Evaluations

This example code produces 4 evaluations:

###Synchronic Evaluation

The synchronic evaluation essentially measures how much the word vectors reflect human intuitions. Essentially, you pick a fixed point of time (we use mid 1995) and run a standard word vector evaluation on the vectors from that time (we use the MEN task)

###Synthetic Evaluation

###Nearest Neighbor Evaluation

###Speed graphs




##Citation

If you found this helpful, please consider citing our paper:

    @inproceedings{rosenfeld-erk-2018-deep,
        title = "Deep Neural Models of Semantic Shift",
        author = "Rosenfeld, Alex  and
          Erk, Katrin",
        booktitle = "Proceedings of the 2018 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers)",
        month = jun,
        year = "2018",
        address = "New Orleans, Louisiana",
        publisher = "Association for Computational Linguistics",
        url = "https://www.aclweb.org/anthology/N18-1044",
        doi = "10.18653/v1/N18-1044",
        pages = "474--484",
    }
    
    
    
 If you have any questions, feel free to contact <alexbrosenfeld@gmail.com>.