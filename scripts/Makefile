
# run this with make -j 36 all for example so that it uses a decent number of processors.

pcfg_files = $(wildcard ../data/test*/*.pcfg)
json_files = $(patsubst %.pcfg,%.json,$(pcfgfiles))
srcdir = "../locallearner"


List of file types

.pcfg : grammar to be learned
.trees : training data trees
.strings : training data strings
.mlpcfg : ML grammar reestimated from the trees



%.wcfg : %.strings
	python ${srcdir}/run_quick_learner.py  --seed 1 --nonterminals 10 --skipio  --min_count_nmf 1000 --number_clusters 25  $<  $@

# %.mjio : %.wcfg
# 	python ../testpcfg/convert_wcfg_to_mjio.py $<  $@

%.strings : %.trees
	python ${srcdir}/convert_trees_to_yields.py $<  $@

%.trees : %.pcfg
	python ${srcdir}/sample_corpus.py --omitprobs --seed 1 --n 1000000  $<  $@

%.mlpcfg : %.trees
	python ${srcdir}/convert_trees_to_pcfg.py  $<  $@

%.mjio : %.wcfg
	python ${srcdir}/convert_wcfg_to_mjio.py $<  $@


%.cmjio : %.mjio %.strings 
	../bin/io -p 1e-6 -d 1000 -g $^  > $@

%.cpcfg : %.cmjio
	python ${srcdir}/convert_mjio_to_pcfg.py $<  $@



%.json :  %.pcfg %.wcfg
	python ${srcdir}/evaluate.py   --json $@  $^


json : $(json_files)

clean : 
	rm $(json_files)

.PHONY : json clean

# So intermediate files are preserved

.SECONDARY :
#.PRECIOUS: $(wcfg_files) $(json_files) $(cpcfg_files)