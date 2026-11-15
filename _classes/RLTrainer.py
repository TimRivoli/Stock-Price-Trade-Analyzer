# File: RLTrainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from typing import List, Tuple
from _classes.RLEnvironment import FILTER_BLENDS, STOCK_COUNTS


def _match_filter_to_blend(filters) -> int:
    """Map an AdaptiveConvex filter list to the nearest FILTER_BLENDS key (Jaccard similarity)."""
    if not filters:
        return 1
    fset = {f[0] if isinstance(f, tuple) else int(f) for f in filters}
    best_key, best_score = 1, -1.0
    for key, blend in FILTER_BLENDS.items():
        if blend is None:
            continue
        bset = {f[0] if isinstance(f, tuple) else int(f) for f in blend}
        score = len(fset & bset) / max(len(fset | bset), 1)
        if score > best_score:
            best_score, best_key = score, key
    return best_key


def _match_count_to_idx(stock_count: int) -> int:
    """Return the index in STOCK_COUNTS nearest to stock_count."""
    return min(range(len(STOCK_COUNTS)), key=lambda i: abs(STOCK_COUNTS[i] - stock_count))

# ─────────────────────────────────────────────
# Phase 1: Behavioral Cloning (Supervised Warmup)
# ─────────────────────────────────────────────
# 
# The key insight: instead of argmax labels (what failed before),
# we use SOFT TARGETS — the full distribution of action values
# that AdaptiveConvex would have chosen.
#
# For each historical date, we run AdaptiveConvex and record:
#   - Which filter blend it selected (GetExecutionFilters)
#   - Which stock count it chose (GetStockCount)
#   - What conviction_score was (→ confidence weight for the label)
#
# We then train the actor to match these soft-weighted decisions
# using cross-entropy loss weighted by conviction_score.
# High-confidence decisions get stronger gradients; uncertain days
# have little influence.

def generate_behavioral_cloning_data(
    picker: 'StockPicker',
    price_dates: list,
    universe_size: int = 200
) -> List[Tuple[np.ndarray, int, float]]:
    """
    Generate (state, action, weight) tuples from historical AdaptiveConvex decisions.
    Weight = conviction_score (not a hard label — a confidence-weighted soft target).
    """
    data = []
    for date in price_dates:
        try:
            ms = picker._get_market_state_smoothed(date, universe_size)
            if ms is None:
                continue
            
            # Translate AdaptiveConvex's natural choices to action indices
            filters = ms.GetExecutionFilters()
            stock_count = ms.GetStockCount()
            conviction = ms.conviction_score
            
            # Map filter list → nearest blend key
            blend_key = _match_filter_to_blend(filters)
            count_idx = _match_count_to_idx(stock_count)
            action_idx = blend_key * 2 + 1  # rebalance=True; stock count fixed at 9
            
            # We'll need to build state from env — skip here and collect via env.reset/step
            data.append((date, action_idx, conviction))
        except Exception:
            continue
    return data


def behavioral_cloning_loss(
    model: 'MomentumActorCritic',
    states: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss, weighted by conviction_score.
    High-conviction historical decisions get stronger gradient signal.
    This avoids the argmax problem: uncertain days (weight ≈ 0.3) have
    ~3x less influence than high-conviction days (weight ≈ 0.9).
    """
    logits, _ = model(states)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    ce = -log_probs[range(len(actions)), actions]   # Per-sample cross-entropy
    return (ce * weights).mean()


# ─────────────────────────────────────────────
# Phase 2: PPO Online Training
# ─────────────────────────────────────────────

class PPOTrainer:
    def __init__(
        self,
        model: 'MomentumActorCritic',
        lr: float = 3e-4,
        gamma: float = 0.99,          # Discount; gamma^50 ≈ 0.60 at 5-day steps (~1-year horizon)
        gae_lambda: float = 0.95,     # GAE smoothing
        clip_epsilon: float = 0.2,    # PPO clip ratio
        entropy_coef: float = 0.10,   # Initial exploration bonus; decayed externally per pass
        value_coef: float = 0.5,      # Critic loss weight
        n_epochs: int = 10,           # PPO update epochs per rollout
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.n_epochs = n_epochs
        
        # Tracking
        self.episode_returns = deque(maxlen=100)
        self.episode_cagrs = deque(maxlen=100)
        # Welford running stats for online reward normalisation
        self._rn_count = 0
        self._rn_mean  = 0.0
        self._rn_m2    = 0.0
    
    def compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float,
    ) -> Tuple[List[float], List[float]]:
        """Generalized Advantage Estimation. Reduces variance vs raw returns."""
        advantages = []
        returns = []
        gae = 0.0
        
        for i in reversed(range(len(rewards))):
            if dones[i]:
                next_val = 0.0
            else:
                next_val = values[i + 1] if i + 1 < len(values) else next_value
            delta = rewards[i] + self.gamma * next_val - values[i]
            gae = delta + self.gamma * self.gae_lambda * (0 if dones[i] else gae)
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
        
        return advantages, returns
    
    def _reset_nan_weights(self):
        for p in self.model.parameters():
            if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                p.data = torch.nan_to_num(p.data, nan=0.0, posinf=1.0, neginf=-1.0)

    def update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        old_log_probs: torch.Tensor,
        returns: torch.Tensor,
        advantages: torch.Tensor,
    ) -> dict:
        """One PPO update pass over collected rollout."""
        states = torch.nan_to_num(states, nan=0.0, posinf=1.0, neginf=-1.0)
        # Normalize advantages (critical for stability)
        if advantages.numel() > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = torch.nan_to_num(advantages, nan=0.0)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0

        for _ in range(self.n_epochs):
            # Forward pass
            logits, values = self.model(states)
            logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
            dist = torch.distributions.Categorical(logits=logits)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            # PPO clipped policy loss
            ratio = torch.exp(new_log_probs - old_log_probs)
            ratio = torch.clamp(ratio, 0.0, 10.0)  # prevent exp overflow
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss (MSE between predicted and actual returns)
            value_loss = nn.functional.mse_loss(values.view(-1), returns.view(-1))

            # Total loss
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

            if not torch.isfinite(loss):
                continue

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            self._reset_nan_weights()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
        
        return {
            "policy_loss": total_policy_loss / self.n_epochs,
            "value_loss": total_value_loss / self.n_epochs,
            "entropy": total_entropy / self.n_epochs,
        }
    
    def _update_reward_stats(self, r: float):
        """Welford online algorithm — update running mean and variance."""
        self._rn_count += 1
        delta = r - self._rn_mean
        self._rn_mean += delta / self._rn_count
        self._rn_m2   += delta * (r - self._rn_mean)

    def _normalize_reward(self, r: float) -> float:
        """Standardise reward using running stats; returns r unchanged until count >= 2."""
        if self._rn_count < 2:
            return r
        std = (self._rn_m2 / (self._rn_count - 1)) ** 0.5
        return (r - self._rn_mean) / max(std, 1e-8)

    def run_episode(self, env: 'RLTradingEnvironment') -> dict:
        """Collect one full episode (one year of trading) of experience."""
        state = env.reset()
        states, actions, log_probs, rewards, values, dones = [], [], [], [], [], []

        done = False
        episode_return = 0.0

        while not done:
            state = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
            state_t = torch.FloatTensor(state).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action(state_t)

            next_state, reward, done, info = env.step(action.item())
            self._update_reward_stats(reward)
            reward = self._normalize_reward(reward)

            states.append(state)
            actions.append(action.item())
            log_probs.append(log_prob.item())
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            episode_return += reward

            state = next_state
        
        # Compute GAE
        next_value = 0.0
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Convert to tensors — sanitize everything before update
        states_arr = np.nan_to_num(np.array(states, dtype=np.float32), nan=0.0, posinf=1.0, neginf=-1.0)
        log_probs   = [0.0 if not np.isfinite(lp) else lp for lp in log_probs]
        rewards     = [0.0 if not np.isfinite(r)  else r  for r  in rewards]
        advantages  = [0.0 if not np.isfinite(a)  else a  for a  in advantages]
        returns_raw = [0.0 if not np.isfinite(r)  else r  for r  in returns]
        states_t        = torch.FloatTensor(states_arr)
        actions_t       = torch.LongTensor(actions)
        old_log_probs_t = torch.FloatTensor(log_probs)
        returns_t       = torch.FloatTensor(returns_raw)
        advantages_t    = torch.FloatTensor(advantages)
        
        # PPO update
        update_stats = self.update(states_t, actions_t, old_log_probs_t, returns_t, advantages_t)
        
        # Episode summary
        self.episode_returns.append(episode_return)
        return {
            **update_stats,
            "episode_return": episode_return,
            "final_portfolio_value": info.get("portfolio_value", 0),
            "n_steps": len(rewards),
        }

    def bc_warmup(self, env: 'RLTradingEnvironment') -> dict:
        """
        One supervised pass through the episode using AdaptiveConvex decisions as
        teacher labels.  Called once on a model cold-start before PPO training begins,
        giving the network a sensible starting policy instead of random exploration.
        """
        state = env.reset()
        done  = False
        total_loss, n = 0.0, 0
        while not done:
            ms = env.picker._get_market_state_smoothed(
                env.tm.currentDate, len(env.picker._tickerList)
            )
            if ms is not None:
                blend_key     = _match_filter_to_blend(ms.GetExecutionFilters())
                conviction    = float(ms.conviction_score)
                rebalance_int = 1 if conviction > 0.6 else 0
                teacher       = blend_key * 2 + rebalance_int  # matches 16-action space
                weight        = conviction

                state_arr = np.nan_to_num(np.asarray(state, dtype=np.float32), nan=0.0)
                loss = behavioral_cloning_loss(
                    self.model,
                    torch.FloatTensor(state_arr).unsqueeze(0),
                    torch.LongTensor([teacher]),
                    torch.tensor([weight]),
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                self._reset_nan_weights()
                total_loss += loss.item()
                n += 1
                state, _, done, _ = env.step(teacher)
            else:
                state, _, done, _ = env.step(0)
        return {"bc_loss": total_loss / max(n, 1), "bc_samples": n}