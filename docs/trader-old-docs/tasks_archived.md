- [x] cross assets tool doesn't work properly
	- not ulizing config mechanisms cli as we have ie scraming about seed being -1...
	- also i wanted it to run remotely, or let's have a param for it, also utilinzg vastrun ai sh
- [x] default configs not logged
	- i want all config values logged properly, i'm worried that when something not in yaml and default value used this default value won't be logged to wandb, fix if i'm correct
- [x] Claude eksperymentator

- [x] ~~when val on and early stopp~~ — **Already correct**: checkpoint saved only on val_loss improvement (`standard.py:335`), eval loads from disk (`stage1_eval.py:113`), so best-val-loss model is always used
- [x] lr_scheduler should be configurable to other scheduler also
- [x] changing max rows to something causes error...
- [x] shortcuts => cause was running ubuntu on xorg
	- [x] in ubuntu pycharm shortcut ctrl+shift+N doesn't work why do you think? maybe some conflict with gnome or soommething shortcuts? can we turn them off ?
	- [x] on laptop buy regular pycharm license, eap even shortcuts doestn' work (keymap)
- [x] guake tab: DON'T CLOSE AFTER LIMIT RESET RUN LAST PROMPT AGAIN
- [x] offline paremeter to have offline wandb
- [x] do smaller run to be role model run
- [x] discrapences in sleek and newer, we can just download both models from both runs from wandb and test inferences from them, and run multiple local tests to find out why evals give differnet resutls
- [x] mamy overfit teraz na jakich danych z ic markets > download 20 with best volume
- [x] are we sure segment id is not passed to the model so it won't know about it?
- [x] vastai.sh to commit also — when running remote it doesn't commit or not locally, let's change it, maybe we need sh runner, make it super simple, minimal changes of code, should be commited locally and tagged locally (so some com between remote and local process controlnig should be), figure out the simplest solution, minimal code
- [x] IN_PROGRESS big refactoring
- [x] IN_PROGRESS in tests dropout for 1 layer
	- tests/test_gru_models.py::TestBuildPredictor::test_unknown_arch_returns_gru
  /home/szmeku/projects/trader/.venv/lib/python3.13/site-packages/torch/nn/modules/rnn.py:123: UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.2 and num_layers=1
	    it still occurs, are our functions ignoring paremeters? (dropout?) i thought we fixed this test, check if it's some bigger issue
- [x] run.py test removed (was identical to `run.py train`), config.test.yml deleted
- [x] full pipelnie diagram
	- i want to get full ML Pipeline diagram, e2e, ie in it when normalization happen when unormalization, when metrics calced etc, full flow e2e in smiple manner. you can use some diagram language that most md readers rendre by default. Check my code extensively so it's factual.
- [x] Fix rolemodel test after simp_metrics update (ratios+diffs, percentile-matched baselines)
	- run old with more training examples since evals are poor because of 4 train samples
- [x] new/additional simplified trading evaluation
	- let's add new fast eval metrics. Simplified trading.
	Baseline will be just sum over each frame of real log_return based on truth, earn both on short and long
		additional baselines just short and just long
	Pred result will be similar to baseline but decision to enter or not (count or not) based on pred but counted by truth, and again combined short and long
		and additionaly also seprate short and long
	Should work for differnt horizons and loockbacks (just taking from config what was set), log returns are additive so should be simple right ?
	Additionaly we want ratios of pred_metric/baseline
	Also let's add sub metrics by percentiles (for non-baseline of course percentile based on pred, how high or low, not truth, no future leakage)
	All metrics should have prefix simp_ or val__simp_
	no fees, no spread (we're doing simple now)
	tell me now if you have any questions
	- when having predictions, act on it on each frame and count return, no fee, sum returns and compare with truth with the same algorithm
	- [x] percentile ratios are not based on percentile baseline.. should be fixed, basing them on regular baseline is useless
- [x] [[remote results differnt.. precision-2]]
	- something is wrong with remote execution running rolemodel config remotely gettin gdifferent results, maybe by default torch or something has setup differnt precision ? let's make it exactly the same as local. local runs rich-cosmos-747 gentle-mountain-707(role model) exaclty the same. remote gallant-river-743 has different results. you have tools/diff tool for comparing runs results
- [x] [[modify vast ai to sort by DLPERF, also allow multi gpu instances]]
- [x] [[embeddings.py — does batch processing but that's for inference, not training data loading]]
	- do we have 2 codes doing similar for things for differnet purposes? why not consolidate it. Remove redundancy
- [x] same speed after superbatch implementation?
- [x] i think in our simp memtrics we're not counting losses
	- because somehow we have
	-  stage1/eval/dir_acc_p75: 0.48984937786509497
	- stage1/eval/simp_p75_long: 5.506900787353516
	- check if we decide based on prediction
	- also if we takie p75 based on prediction (not truth)
	- if we count losses in simp, (so they decrease our profit)
	- what's the bankrupcy treshold?
	- let's add sharpe ratio and sortino to simp metrics
- [x] logretModel/logretOracle wyszlo mi jako najlepszy wskaznik
	- [x] #IN_PROGRESS checking if our metrics docs uses same definitions as code
	- sortino with holding benchmark (whole period long or short ) dolna granica
	- logrets raio wskazuje gorna imposiblowa granice (oraclowa)
	- what if we could come up with loss that combines these 2
- [x] MT check if we can download manually 3 years of btc
	- on my account I can't, waitnig for Kristof for his feedback
- [x] c0.yml veri good, continue
	- na wiekszym secie trening
		- czy loss super maly
		- jak simpy wypadly
