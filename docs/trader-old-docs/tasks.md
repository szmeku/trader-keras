## March 18
- refactor do lightning
	- [x] goal reducing code base without changing how it works so rolemodel test should work unchanged 
	- splitting to packages would reduce it or not ? 
	- focusing on small atomic things (tests etc )
- overfit RL continue
	- wg https://chatgpt.com/c/69b9d293-a9b4-8327-b75f-a6f7f1965320
- smaller later
	- no val on train_ratio=1
- trading env 

## Rest 


- [ ] multi loss 
- [ ] new return loss check it, test it
- [ ] training huber then sum log returns or just huber
	- check how switching loss can finetune, or using it even more dynamically (every n epochs)
- [ ] cos sie powaznie zdupczylo z symulatorem nie pasuja wyniki do mt5
  - kontynuowc limitowane instancje claude'a
  - przepiecie na konto demo nie pomoglo
- [ ] we can improve further envs, let's leave for later
  - for mt5 real ticks one year of btc differnces bigger
- [ ] overfit some nn vs mt and local.
  - new oracle RL baseline, let's train RL xgboost to basically remember perfectly what to trade when to get perfect results. MAXIMUM TOP WE can earn
    - we have simulator env (adjusted for RL)
    - we should have code just for evaluation in our simulation (usable by tools/validate)
    - 
    - then we'll do dll from it and we'll run it in mt5 to validate we have the same (dll simplest solution) 
    - we should have it worknig for any assets   
  - overfit btc on high low and log return
  - what's the target now?	 	
    - convert to ddl (onnx or something), run simulated trading on it compare to env
      - result on 
- [ ] ENVSIMULATION (first investigate and let's discuss)
  - [ ] if when running strategy tester we can't use http requests let's test saved (params + time) -> action mappings that will be hardcoded in some file and read by strategy instead of http requestsale na ale    
  - [ ] envsim let's check other ranges and assets so we're sure that comparison of envs (our sim vs mt5 strategy tester) works 
  - [ ] let's make sure it's gonna work for other strategies as well so we don't have some weird hacks
  - [ ] test delay
    - sim config vs mt tester 
  - [ ] should we improve speed for RL ? or not

- [ ] mt, odpalanie testu przykladowej strategii 
- [ ] [[ docs/env_specs ]]
- [ ] train loss goes up and val goes down, almost immediately
	- [ ] check in old setups if we have the same
		- where loss super low, data params ratio ok, not many total epochs

- [ ] expanding knowledge
	- [ ] what is pearson ? 
	- [ ] how adamW works and weights decay
	- [ ] multitime step, attention seq2seq, are we using it if not why?
- [ ] sprawdzic wyniki testowania nowych config params configi test_*
	- [ ] are they telling us anything? 
	- [ ] should be repepated on bigger sets? 
- [ ] is attention for sure implemented correctly
- [ ] understand differenrce multiple timesteps vs not? multi => attention, seq2seq
- [ ] w ciul nowych danych w ~/data.zip
	- [ ] przekonwertowac do parquetow
	- [ ] new cross asset checking
		- new big files
		- without last 3 months
		- should remotely
- [ ] when can't avoid overfit regularization
	- l1,l2,dropout what else
- [ ] claude experimentator improvement
	- ask smart llms about better prompt
- [ ] early data analysis
- [ ] long big file data loading
	- === Stage 1: Training GRU Predictor ===
		  Seed: 981432709 (deterministic mode)
		Loading data from /home/szmeku/projects/data (1 file(s))...
		  binance_aaveusdt_20241012_0000_to_20251011_2359.parquet: 9,803,967 bars (before dropna)
		After features + dropna: 9,803,907 rows, 1 segment(s)
		Train: 9,705,867, Val: 98,040
		Stored in RAM
- [ ] wb & optuna
  - local controller uses optuna for next params
  - [ ] check gpt conversation what to remember about, ie nstart trials for good samples of results for optuna
  - when agent not running anything for 5min should destroy itself
- [[smaller batches]]
	- does it make sens to have small batches (will need less vram), faster convergence? what literature says
	- let's experiment, 5 small assets (for fast local tests), 10 runs with different batchs
- [ ] multi-asset training by classified by smarter data properties
  - categorization automated by props of data (not manual ie akcje, indexy)
- [ ] LR z xgboost na bazie statow wygenerowanych przez gru
- [ ] LR Lorą
- [ ] proper generalization
  - [ ] work foward optimization wfo
- [[test chronos2 finetuning]]
- [ ] co to jest rezim? (dokladnie), przewidywanie zmian rezimu
- [ ] how often are we updating thethas? maybe we should do it more often?
- [ ] mamy overfit teraz na jakich danych z ic markets
	- [[70assets with highest std run on 20k epochs]]
	- [[70assets highest std but for , running 20k epochs]]
	- [ ] overfit on one
	- pomysly
		- live tickami czy potrafie przewidywac minuty kolejne
		- czy w ogole mi to cos daje
		- co to jest zmiana rezimu jak ja wykryc
		- wybrac assety po tickach
		- gru for trend some anomaly detection for max,min ...
- [ ] overfit not done, if we know future perfectly we should be able to play on it perfectly, super profit and so on. prove on overfit model
	- more features more targets
	- from what features what targets
- [ ] continuing simulation: but of course in this simul we should never use data from the future, in real playing we can't do an ordre with future data we awnt to be realistic
- [ ] memorizing multi asset
	- [ ] for the successful one asset d/p ratio, try to memorize multiasset
- [ ] continuity
	- are we for sure not shuffling frames in one sequence ? we can shuffle whole batches with differnet segments right to show variety of differnt data to model quickly but we shouldn't break continuity in one batch right ?
- [ ] protect from mixing frames
	- how is resampling working in our code if i have few parquet files with different assets, i don't wnt them to be mixed up right, but also i don't want to pass any information that they are different assets, we're trying to extract general knowledge about price movements model should not know they're comming from difrernet assets but still we don't want to mixed in one batch from multiple assets right?
- [ ] differnet strides horizons and params/data ratio experiment, sleek-dust, upbeat and one later
- [ ] run exepriments
- [ ] refactor in interwined
	- we should check pluggability maintanability patterns/design in our code ie, normalization and denormalization should leave together right? they are dependend interconnected, we don't want to change on without changing the other, it's basically the same process but reversed right? so we shlule be able to change normalizer or add new with it's denormalizer together at once, find more interwinded stuff like this and make it easier to maintain switch, extend etc..
- [ ] data now is super dependent, let's preapre stride > lookback
- [ ] wandb agenets for sweeping
- [ ] use 3rd part libraries to not maintani so much code
- [ ] should we use mls?
- [ ] while using transformer doesn't utilize alll the memory, now just 50%
- [ ] nn now is too big, we don't need so many params for such a small data, should be more less equall right?
- [ ] action time 0-1day (because of rollover fee on ic markets
- [[docs/task_overfit]]


# notes
1. Stride is the biggest lever: stride=100 (1.54e-14) > stride=50 (2.09e-14) > stride=30 (6.15e-14) > stride=10 (1.39e-13).
  More stride = fewer samples = easier memorization
  2. OneCycle works well: max_lr=0.08 (4.16e-14) and max_lr=0.05 (4.51e-14) — competitive with plateau
  3. h=64 L=2 (7.23e-14) and h=64 L=3 + stride=30 (6.54e-14) — bigger models still good
  4. Plateau patience=15 is optimal: p=10 (1.75e-13), p=20 (6.8e-9), p=25 (9e-10) — patience>15 hurts badly!
  5. factor=0.3 (1.15e-13) works, factor=0.7 (4.34e-7) too gentle — 0.5 is the sweet spot
  6. clip=10 + stride=10 (9.18e-14) — clip=10 works. clip=3 (1.89e-11), clip=2 (3.11e-6) too tight
  7. lr=0.07 (3.86e-13) OK, lr=0.06 (2.61e-8) oddly bad, lr=0.04 (6.15e-13) OK
  8. lookback=300 (1.34e-13) fine, lookback=800 (4.63e-11) slightly worse
  9. OneCycle + h=64 L=3 (#30: 2.02 — DIVERGED!) — bad combo, avoid
  10. h=50 L=2 (3.41e-6) and h=48 L=3 (4.53e-8) — oddly worse than h=30 or h=64

# [[tasks_archived]]
# [[docs/knowledge]]
