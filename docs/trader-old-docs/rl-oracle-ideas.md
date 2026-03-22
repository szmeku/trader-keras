# Oracle & RL Algorithm Ideas (2026-03-14)

## Oracle Benchmark: RL Agent with Future Leak

The oracle is a **regular RL agent that cheats** — it sees future bars as extra features. Trains through TradingEnv like any other agent. Same action space, same rewards, same rules. The env stays clean; the cheating happens agent-side.

**Old approach (DP solver) — deleted:** Had its own inline cost model that diverged from the sim (no rollover, no stopout, no margin). Replaced by this RL approach.

**How it works:**
1. Oracle agent gets the full dataset at construction time
2. On each `env.step()`, agent receives standard obs from env
3. Agent internally looks up future bars (next N) by current bar index
4. Concatenates future data with obs → feeds to its policy network
5. Train DQN/PPO until convergence — signal is extremely strong with perfect foresight
6. Result: benchmark P&L = the ceiling any real agent tries to approach

**Why it converges fast:**
- Future knowledge makes the problem trivially learnable
- State space is small (21 obs floats + future bar features)
- Single episode (50K bars) — small dataset
- No exploration needed — the signal is clear

---

## Real RL Algorithms (through TradingEnv)

For agents that learn **without future knowledge** — same interface, same env:

### PPO (recommended first)
- Standard on-policy, handles `Dict(type=Discrete(6), params=Box(2,))` natively
- Stable, well-understood, battle-tested
- Needs many env steps (vectorization helps later)

### DQN
- Off-policy, sample-efficient, works with single env
- Needs discrete action space (simplify to flat/long/short at fixed volumes)

### SAC (Soft Actor-Critic)
- Off-policy, handles continuous actions, entropy bonus for exploration
- More complex but sample-efficient

---

## Architecture Principle

All agents (oracle and real) go through `TradingEnv`:
- Env = MT5 equivalent (raw state: account + market + position + time)
- Agent = EA equivalent (computes features, picks actions)
- Oracle agent just happens to have future data as extra features

```python
obs, info = env.reset()
while not done:
    action = agent.act(obs, info)  # oracle agent peeks at future here
    obs, reward, term, trunc, info = env.step(action)
```
