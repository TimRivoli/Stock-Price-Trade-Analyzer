import torch
import torch.nn as nn

class MomentumActorCritic(nn.Module):
    """
    Separate actor and critic trunks eliminate conflicting gradients.
    Action space: 16 (8 blends × 2 rebalance/hold); stock count fixed at 9.
    """
    def __init__(self, state_dim: int = 31, action_dim: int = 16):
        super().__init__()

        def _trunk():
            return nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
            )

        self.actor_trunk  = _trunk()
        self.actor_head   = nn.Linear(64, action_dim)
        self.critic_trunk = _trunk()
        self.critic_head  = nn.Linear(64, 1)

    def forward(self, state: torch.Tensor):
        logits = self.actor_head(self.actor_trunk(state))
        value  = self.critic_head(self.critic_trunk(state))
        return logits, value

    def get_action(self, state: torch.Tensor, deterministic: bool = False):
        logits, value = self(state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Categorical(logits=logits)
        action = logits.argmax(dim=-1) if deterministic else dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value
