
# run this with make -j 36 all for example so that it uses a decent number of processors.
datadirs = ../data/test*
pcfg_files = $(wildcard $(datadirs)/*.pcfg)
json_files = $(patsubst %.pcfg,%.json,$(wildcard $(datadirs)/*.pcfg))
kernel_files = $(patsubst %.pcfg,%.kernels,$(wildcard $(datadirs)/*.pcfg))
srcdir = ../locallearner

maxlength = 10
#suffixes = json kjson strings trees wcfg 0pcfg 1pcfg 

# List of file types

# .pcfg : grammar to be learned
# .trees : training data trees
# .strings : training data strings
# .mlpcfg : ML grammar reestimated from the trees
# .kernels : json file with the kernels in
# .kjson : evaluation of kernels.




# %.wcfg : %.strings
# 	python ${srcdir}/run_quick_learner.py  --seed 1 --nonterminals 10 --skipio  --min_count_nmf 1000 --number_clusters 25  $<  $@

# %.mjio : %.wcfg
# 	python ../testpcfg/convert_wcfg_to_mjio.py $<  $@

%.strings : %.trees
	python ${srcdir}/convert_trees_to_yields.py $<  $@

%.trees : %.pcfg
	python ${srcdir}/sample_corpus.py --omitprobs --seed 1 --n 1000000  $<  $@

%.mlpcfg : %.trees
	python ${srcdir}/convert_trees_to_pcfg.py --length $(maxlength) $<  $@

%.0mjio : %.0pcfg
	python ${srcdir}/convert_pcfg_to_mjio.py $<  $@


%.1mjio : %.0mjio %.strings 
	../bin/io -n 1 -l $(maxlength) -p 1e-8 -d 1000 -g $^  > $@

%.1pcfg : %.1mjio
	python ${srcdir}/convert_mjio_to_pcfg.py $<  $@


%.final_mjio : %.1mjio %.strings 
	../bin/io -n 100 -l $(maxlength) -p 1e-8 -d 1000 -g $^  > $@

%.final_pcfg : %.final_mjio
	python ${srcdir}/convert_mjio_to_pcfg.py $<  $@

%.kernels : %.strings
	python ${srcdir}/run_find_kernels.py --min_count_nmf 1000  --cheat $*.pcfg $< $@

%.kjson :  %.kernels
	python ${srcdir}/evaluate_kernels.py  --json $@  $*.pcfg $< 


%.wcfg : %.strings %.kernels
	python ${srcdir}/run_make_wcfg.py $^ $@

%.0pcfg : %.wcfg
	python ${srcdir}/convert_wcfg_to_pcfg.py $<  $@

%.json :  %.mlpcfg %.1pcfg %.final_pcfg
	python ${srcdir}/evaluate_pcfg.py --maxlength $(maxlength)  --json $@  --target $*.pcfg $^ 


json : $(json_files)

# allkernels : $(kernel_files)

clean : 
	rm -f $(datadirs)/*.{json,kjson,strings,trees,wcfg,kernels,0pcfg,1pcfg,final_pcfg,final_mjio,mlpcfg,0mjio,1mjio}

.PHONY : json clean 

# So intermediate files are preserved

.SECONDARY :

#.SECONDARY : %.1mjio %.0mjio %.kjson %.kernels %.strings %.trees %.wcfg %.mlpcfg %.0pcfg %.1pcfg  %.json