
# run this with 
# make -j 50 json 
# for example so that it uses a decent number of processors.
datadirs = ../data/test*
pcfg_files = $(wildcard $(datadirs)/*.pcfg)
json_files = $(patsubst %.pcfg,%.json,$(wildcard $(datadirs)/*.pcfg))
kernel_files = $(patsubst %.pcfg,%.kernels,$(wildcard $(datadirs)/*.pcfg))
srcdir = ../locallearner

maxlength = 10
evallength = 20
sgd_epochs = 1
sgd_lr = 0.1
sgd_batchsize = 10000

#suffixes = json kjson strings trees wcfg 0pcfg 1pcfg 

# List of file types

# .pcfg : grammar to be learned
# .trees : training data trees
# .strings : training data strings
# .mlpcfg : ML grammar reestimated from the trees
# .kernels : json file with the kernels in
# .kjson : evaluation of kernels.
# .sgd_pcfg : grammar trained with SGD (exponentiated gradient via IO)




# %.wcfg : %.strings
# 	python ${srcdir}/run_quick_learner.py  --seed 1 --nonterminals 10 --skipio  --min_count_nmf 1000 --number_clusters 25  $<  $@

# %.mjio : %.wcfg
# 	python ../testpcfg/convert_wcfg_to_mjio.py $<  $@

%.strings : %.trees
	python ${srcdir}/convert_trees_to_yields.py $<  $@ >> $*.log

%.trees : %.pcfg
	python ${srcdir}/sample_corpus.py --omitprobs --seed 1 --n 1000000  $<  $@ >> $*.log

%.sigma : %.pcfg
	python ${srcdir}/save_terminals.py $<  $@ >> $*.log

%.mlpcfg : %.trees
	python ${srcdir}/convert_trees_to_pcfg.py --length $(maxlength) $<  $@ >> $*.log

%.0mjio : %.0pcfg
	python ${srcdir}/convert_pcfg_to_mjio.py $<  $@ >> $*.log


%.1mjio : %.0mjio %.strings 
	../bin/io -n 1 -l $(maxlength) -p 1e-8 -d 1000 -g $^  > $@ 2>> $*.log

%.1pcfg : %.1mjio
	python ${srcdir}/convert_mjio_to_pcfg.py $<  $@ >> $*.log


%.final_mjio : %.1mjio %.strings 
	../bin/io -n 100 -l $(maxlength) -p 1e-8 -d 1000 -g $^  > $@ 2>> $*.log

%.final_pcfg : %.final_mjio
	python ${srcdir}/convert_mjio_to_pcfg.py $<  $@ >> $*.log

%.kernels : %.strings
	python ${srcdir}/run_find_kernels.py --min_count_nmf 1000  --cheat $*.pcfg $< $@ >> $*.log

%.kjson :  %.kernels
	python ${srcdir}/evaluate_kernels.py  --json $@  $*.pcfg $< >> $*.log


%.wcfg : %.strings %.kernels
	python ${srcdir}/run_make_wcfg.py $^ $@ >> $*.log

%.0pcfg : %.wcfg
	python ${srcdir}/convert_wcfg_to_pcfg.py $<  $@ >> $*.log

%.sgd_pcfg : %.0pcfg %.strings
	python ${srcdir}/run_sgd_io.py --io ../bin/io --maxlength $(maxlength) --epochs $(sgd_epochs) --lr $(sgd_lr) --batchsize $(sgd_batchsize) --shuffle $< $*.strings $@ >> $*.log 2>&1

%.json :  %.mlpcfg %.0pcfg %.1pcfg %.final_pcfg %.sgd_pcfg
	python ${srcdir}/evaluate_pcfg.py --maxlength $(evallength)  --json $@  --target $*.pcfg $^  >> $*.log


json : $(json_files)

# allkernels : $(kernel_files)

clean : 
	rm -f $(datadirs)/*.{log,json,kjson,strings,trees,wcfg,kernels,0pcfg,1pcfg,final_pcfg,final_mjio,mlpcfg,0mjio,1mjio,sgd_pcfg}

thin :
	rm -f $(datadirs)/*.{strings,trees,final_mjio,0mjio,1mjio}	 

.PHONY : json clean thin

# So intermediate files are preserved

.SECONDARY :

#.SECONDARY : %.1mjio %.0mjio %.kjson %.kernels %.strings %.trees %.wcfg %.mlpcfg %.0pcfg %.1pcfg  %.json