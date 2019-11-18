Supplementary materials for synthetic evaluation


### time_independent_words.txt ###

This is a list of all words in the English Fiction portion of the Google Books Ngram (Lin et al. 2012) corpus that appear in the top 20,000 by frequency in every decade as calculated by Hamilton et al. (2016).


### BLESS_classes.txt ###

This contains the words in each BLESS class (Baroni et al. 2011) that appear in time_independent_words.txt. Each line is the name of the BLESS class, a tab, then a space separated list of words in that class.


### synthetic_words_set_1.csv ###
### synthetic_words_set_2.csv ###
### synthetic_words_set_3.csv ###

These csvs contain the data for the three sets of 15 synthetic words. See description in main paper for details. r1 and r2 are the component real words that form the synthetic word r1◦r2. s is the steepness of the sigmoid curve that defines how r1◦r2 shifts in meaning from r1 to r2. m is the sigmoid curve's midpoint.


### License ###

This dataset is derived from the BLESS dataset from Marco Baroni and Alessandro Lenci. The BLESS dataset is licensed under the Creative Commons
Attribution-ShareAlike license. As such, the material in our dataset is released under the same license. In short, you are free to copy,
distribute, transmit and edit the data set, as long as you credit the
original authors and, if you
choose to distribute the resulting work, you release it under the same
license.

For more information, see:

http://creativecommons.org/licenses/by-sa/3.0/


### References  ###

Baroni, M. and Lenci, A., 2011, July. How we BLESSed distributional semantic evaluation. In Proceedings of the GEMS 2011 Workshop on GEometrical Models of Natural Language Semantics (pp. 1-10). Association for Computational Linguistics.

William L. Hamilton, Jure Leskovec, and Dan Jurafsky. ACL 2016. Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change.

Lin, Y., Michel, J.B., Aiden, E.L., Orwant, J., Brockman, W. and Petrov, S., 2012, July. Syntactic annotations for the google books ngram corpus. In Proceedings of the ACL 2012 system demonstrations (pp. 169-174). Association for Computational Linguistics.


