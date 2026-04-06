# services/ml-service/models/model_9_rl_agent.py
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.distributions import Normal
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
import logging
import pickle
import hashlib
import random
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
from dataclasses import dataclass
from datetime import datetime

_NNBase = nn.Module if TORCH_AVAILABLE else object

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


@dataclass
class RLConfig:
    """Configuration for RL Policy Agent V2."""
    state_dim: int = 64
    action_dim: int = 1  # Continuous stake percentage
    hidden_dim: int = 256
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    critic_coef: float = 0.5
    entropy_coef_start: float = 0.01
    entropy_coef_end: float = 0.001
    max_grad_norm: float = 0.5
    epochs_per_update: int = 10
    batch_size: int = 64
    target_kl: float = 0.01
    device: Optional[str] = None
    max_stake: float = 0.20
    min_edge_threshold: float = 0.02
    drawdown_limit: float = 0.15
    no_bet_penalty: float = 0.01
    risk_penalty_coef: float = 0.1
    max_time_window_hours: float = 48.0


class Experience:
    """Experience tuple for RL memory."""
    def __init__(self, state, action, reward, next_state, done, log_prob, value):
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
        self.log_prob = log_prob
        self.value = value


class RolloutBuffer:
    """Buffer for storing rollout experiences."""

    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def add(self, experience: Experience):
        self.states.append(experience.state)
        self.actions.append(experience.action)
        self.rewards.append(experience.reward)
        self.next_states.append(experience.next_state)
        self.dones.append(experience.done)
        self.log_probs.append(experience.log_prob)
        self.values.append(experience.value)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

    def __len__(self):
        return len(self.states)


class ContinuousActorCritic(_NNBase):
    """
    Continuous Actor-Critic network for PPO.

    Actor: Outputs mean and log_std for normal distribution
    Critic: Outputs state value
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, max_stake: float = 0.20):
        super().__init__()
        self.max_stake = max_stake

        # Shared backbone
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )

        # Actor head (mean and log_std for continuous action)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1))

        # Critic head
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.orthogonal_(p, gain=np.sqrt(2))

    def forward(self, state):
        shared_features = self.shared(state)
        mean = self.actor_mean(shared_features)
        mean = torch.tanh(mean) * self.max_stake  # Scale to [0, max_stake]
        std = torch.exp(self.actor_log_std).clamp(0.01, 0.5)
        value = self.critic(shared_features)
        return mean, std, value

    def get_action(self, state, deterministic=False):
        """Sample continuous action from policy."""
        mean, std, value = self.forward(state)
        dist = Normal(mean, std)

        if deterministic:
            action = mean
            log_prob = dist.log_prob(action).sum(dim=-1)
        else:
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

        # Clamp action to valid range
        action = torch.clamp(action, 0, self.max_stake)

        return action.squeeze().item(), log_prob, value.squeeze().item()


class SimulatedBettingEnvironment:
    """
    Simulated betting environment for RL training.
    Stateless - bankroll tracked externally.
    """

    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll

    def step(
        self,
        stake_percentage: float,
        confidence: float,
        edge: float,
        odds: float,
        bankroll: float,
        current_drawdown: float
    ) -> Tuple[float, bool, Dict]:
        """
        Execute action and return reward, done, info.

        Args:
            stake_percentage: Percentage of bankroll to stake (0-0.2)
            confidence: Model confidence (0-1)
            edge: Edge vs market (-0.1 to 0.1)
            odds: Decimal odds for the bet
            bankroll: Current bankroll
            current_drawdown: Current drawdown percentage

        Returns:
            reward, done, info
        """
        stake_amount = bankroll * stake_percentage
        profit = 0

        # Determine outcome based on edge and confidence
        # Higher edge and confidence = higher win probability
        win_prob = 0.5 + (edge * 3) + ((confidence - 0.5) * 0.4)
        win_prob = np.clip(win_prob, 0.05, 0.95)

        is_win = np.random.random() < win_prob

        # Risk penalty for large stakes
        risk_penalty = stake_percentage * RLConfig.risk_penalty_coef

        # No-bet penalty when edge exists
        no_bet_penalty = 0
        if stake_amount == 0 and edge > RLConfig.min_edge_threshold:
            no_bet_penalty = RLConfig.no_bet_penalty

        if is_win and stake_amount > 0:
            profit = stake_amount * (odds - 1)
            reward = (profit / self.initial_bankroll) - risk_penalty - no_bet_penalty
        elif stake_amount > 0:
            profit = -stake_amount
            reward = (profit / self.initial_bankroll) - risk_penalty - no_bet_penalty
        else:
            reward = -no_bet_penalty

        # Additional drawdown penalty
        if current_drawdown > RLConfig.drawdown_limit:
            reward -= (current_drawdown - RLConfig.drawdown_limit)

        # Check if episode should end
        new_bankroll = bankroll + profit
        done = new_bankroll <= 0 or new_bankroll >= self.initial_bankroll * 2

        info = {
            "profit": profit,
            "new_bankroll": new_bankroll,
            "won": is_win,
            "win_prob": win_prob,
            "risk_penalty": risk_penalty,
            "no_bet_penalty": no_bet_penalty
        }

        return reward, done, info


class RLPolicyAgent(BaseModel):
    """
    RL Policy Agent V2 - Production-ready betting strategy.

    Fixes applied:
        - Proper GAE implementation
        - Single source of truth for bankroll
        - Continuous action space
        - Kelly Criterion feature
        - Risk-adjusted rewards
        - No-bet penalty
        - KL divergence early stopping
        - Entropy decay
        - Volatility and time-to-match features
        - Sharpe ratio and max drawdown tracking
    """

    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 2,
        params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        epochs_per_update: int = 10,
        batch_size: int = 64,
        max_stake: float = 0.20,
        min_edge_threshold: float = 0.02,
        drawdown_limit: float = 0.15,
        no_bet_penalty: float = 0.01,
        risk_penalty_coef: float = 0.1
    ):
        super().__init__(
            model_name=model_name,
            model_type="RLPolicy",
            weight=weight,
            version=version,
            params=params,
            supported_markets=[
                MarketType.MATCH_ODDS,
                MarketType.OVER_UNDER,
                MarketType.BTTS
            ]
        )

        # RL parameters
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.epochs_per_update = epochs_per_update
        self.batch_size = batch_size
        self.max_stake = max_stake
        self.min_edge_threshold = min_edge_threshold
        self.drawdown_limit = drawdown_limit
        self.no_bet_penalty = no_bet_penalty
        self.risk_penalty_coef = risk_penalty_coef

        # Networks
        device_str = RLConfig.device if RLConfig.device else ('cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu')
        self.device = torch.device(device_str) if TORCH_AVAILABLE else None
        self.actor_critic = ContinuousActorCritic(
            RLConfig.state_dim, RLConfig.hidden_dim, max_stake
        ).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)

        # Buffer
        self.buffer = RolloutBuffer()

        # Environment
        self.env = SimulatedBettingEnvironment()

        # Bankroll tracking (single source of truth)
        self.bankroll: float = 1000.0
        self.initial_bankroll: float = 1000.0
        self.peak_bankroll: float = 1000.0
        self.current_drawdown: float = 0.0

        # Performance tracking
        self.bet_history: List[float] = []
        self.consecutive_losses: int = 0
        self.consecutive_wins: int = 0
        self.total_bets: int = 0
        self.winning_bets: int = 0
        self.total_profit: float = 0.0

        # Training state
        self.trained_episodes: int = 0
        self.best_policy_state: Optional[Dict] = None
        self.best_sharpe: float = -float('inf')

        # Model hash for versioning
        self.model_hash: str = ""
        self._update_model_hash()

        # State cache
        self.last_state: Optional[np.ndarray] = None
        self.last_action: Optional[float] = None

        # Only certify if PyTorch is available
        self.certified = TORCH_AVAILABLE

    def _update_model_hash(self):
        """Update model hash for versioning."""
        model_str = str(self.actor_critic) + str(self.learning_rate) + str(self.version)
        self.model_hash = hashlib.md5(model_str.encode()).hexdigest()[:8]

    def _compute_gae(
        self,
        rewards: List[float],
        values: List[float],
        dones: List[bool],
        next_value: float = 0.0
    ) -> np.ndarray:
        """Proper Generalized Advantage Estimation."""
        advantages = np.zeros(len(rewards))
        gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            else:
                delta = rewards[t] + self.gamma * values[t + 1] * (1 - dones[t]) - values[t]

            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        return advantages

    def _normalize_rewards(self, rewards: List[float]) -> np.ndarray:
        """Normalize rewards for stable training."""
        rewards_arr = np.array(rewards)
        if rewards_arr.std() > 1e-8:
            return (rewards_arr - rewards_arr.mean()) / (rewards_arr.std() + 1e-8)
        return rewards_arr

    def _calculate_kelly(self, confidence: float, odds: float) -> float:
        """Calculate Kelly Criterion stake recommendation."""
        if odds <= 1:
            return 0.0

        b = odds - 1
        p = confidence
        q = 1 - p

        kelly = (b * p - q) / b
        return np.clip(kelly, 0, self.max_stake)

    def _calculate_volatility(self, window: int = 20) -> float:
        """Calculate recent return volatility."""
        if len(self.bet_history) < 5:
            return 0.0

        recent = self.bet_history[-min(window, len(self.bet_history)):]
        return float(np.std(recent))

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio of returns."""
        if len(self.bet_history) < 2:
            return 0.0

        returns = np.array(self.bet_history)
        if returns.std() < 1e-8:
            return 0.0

        return float(returns.mean() / returns.std())

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from bet history."""
        if len(self.bet_history) < 2:
            return 0.0

        cumulative = np.cumsum(self.bet_history)
        peak = np.maximum.accumulate(cumulative)
        drawdowns = (peak - cumulative) / (self.initial_bankroll + peak)

        return float(np.max(drawdowns))

    def _get_entropy_coef(self, episode: int, total_episodes: int) -> float:
        """Decaying entropy coefficient."""
        progress = episode / total_episodes
        coef = RLConfig.entropy_coef_start * (1 - progress) + RLConfig.entropy_coef_end * progress
        return max(RLConfig.entropy_coef_end, coef)

    def _augment_state(
        self,
        base_state: np.ndarray,
        confidence: float,
        edge: float,
        odds: float,
        time_to_match_hours: float = 24.0,
        implied_prob: float = 0.5
    ) -> np.ndarray:
        """Augment state with comprehensive features."""
        state = base_state.copy()

        # Index 0: Normalized bankroll
        state[0] = self.bankroll / self.initial_bankroll

        # Index 1: Consecutive losses (capped)
        state[1] = min(self.consecutive_losses, 10) / 10

        # Index 2: Consecutive wins (capped)
        state[2] = min(self.consecutive_wins, 10) / 10

        # Index 3: Win rate (last 20)
        if self.total_bets > 0:
            recent = self.bet_history[-min(20, len(self.bet_history)):]
            win_rate = sum(1 for p in recent if p > 0) / max(1, len(recent))
            state[3] = win_rate
        else:
            state[3] = 0.5

        # Index 4: Average profit per bet (last 20)
        if self.bet_history:
            recent = self.bet_history[-min(20, len(self.bet_history)):]
            avg_profit = np.mean(recent) / self.initial_bankroll
            state[4] = np.clip(avg_profit + 0.5, 0, 1)
        else:
            state[4] = 0.5

        # Index 5: Total ROI
        if self.initial_bankroll > 0:
            roi = self.total_profit / self.initial_bankroll
            state[5] = np.clip(roi + 0.5, 0, 1)
        else:
            state[5] = 0.5

        # Index 6: Model confidence
        state[6] = confidence

        # Index 7: Edge vs market (properly scaled)
        # edge range: -0.1 to 0.1 -> maps to 0-1
        state[7] = np.clip((edge + 0.1) / 0.2, 0, 1)

        # Index 8: Normalized odds (1-10 -> 0-1)
        state[8] = np.clip((odds - 1) / 9, 0, 1)

        # Index 9: Kelly recommended stake
        kelly = self._calculate_kelly(confidence, odds)
        state[9] = kelly / self.max_stake

        # Index 10: Volatility
        state[10] = self._calculate_volatility()

        # Index 11: Time to match (normalized)
        state[11] = np.clip(time_to_match_hours / RLConfig.max_time_window_hours, 0, 1)

        # Index 12: Market disagreement
        market_disagreement = abs(confidence - implied_prob)
        state[12] = market_disagreement

        # Index 13: Current drawdown
        state[13] = self.current_drawdown

        # Index 14: Sharpe ratio (scaled)
        sharpe = self._calculate_sharpe_ratio()
        state[14] = np.clip(sharpe + 0.5, 0, 1)

        return state

    def _update_policy(self, next_value: float = 0.0):
        """Update policy using proper PPO with GAE."""
        if len(self.buffer) < self.batch_size:
            return

        # Compute proper GAE advantages
        advantages = self._compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.dones,
            next_value
        )

        # Normalize rewards
        normalized_rewards = self._normalize_rewards(self.buffer.rewards)

        # Compute returns
        returns = advantages + np.array(self.buffer.values)

        # Normalize advantages
        if advantages.std() > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.buffer.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer.log_probs)).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)

        # PPO update with early stopping
        for epoch in range(self.epochs_per_update):
            indices = np.random.permutation(len(states))
            epoch_kl = 0

            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]

                # Forward pass
                mean, std, state_values = self.actor_critic(batch_states)
                dist = Normal(mean, std)
                state_values = state_values.squeeze()

                # Compute new log probs
                new_log_probs = dist.log_prob(batch_actions).sum(dim=-1)

                # PPO loss
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)

                policy_loss = -torch.min(
                    ratio * batch_advantages,
                    clipped_ratio * batch_advantages
                ).mean()

                # Value loss
                value_loss = F.mse_loss(state_values, batch_returns)

                # Entropy bonus
                entropy = dist.entropy().mean()

                # Total loss
                loss = policy_loss + RLConfig.critic_coef * value_loss - entropy

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), RLConfig.max_grad_norm)
                self.optimizer.step()

                # Track KL for early stopping
                epoch_kl += (new_log_probs - batch_old_log_probs).mean().item()

            epoch_kl /= max(1, len(states) / self.batch_size)

            # KL early stopping
            if epoch_kl > RLConfig.target_kl:
                logger.debug(f"KL divergence {epoch_kl:.4f} exceeded target, stopping early")
                break

        # Clear buffer
        self.buffer.clear()

    def train(
        self,
        matches: List[Dict[str, Any]],
        episodes: int = 100,
        steps_per_episode: int = 100
    ) -> Dict[str, Any]:
        """Train RL policy agent with proper episode management."""
        if not matches:
            return {"error": "No training data"}

        logger.info(f"Training RL agent on {len(matches)} matches, {episodes} episodes")

        episode_rewards = []
        episode_sharpes = []

        for episode in range(episodes):
            # Reset episode state
            episode_bankroll = self.bankroll
            episode_reward = 0
            episode_steps = 0

            # Shuffle matches for each episode
            shuffled_matches = matches.copy()
            random.shuffle(shuffled_matches)

            # Decaying entropy coefficient
            entropy_coef = self._get_entropy_coef(episode, episodes)

            for step in range(min(steps_per_episode, len(shuffled_matches))):
                match = shuffled_matches[step]

                # Extract features
                confidence = match.get('confidence', 0.6)
                edge = match.get('edge', 0.03)
                odds = match.get('odds', 2.0)
                implied_prob = 1 / odds if odds > 0 else 0.5
                time_to_match = match.get('hours_until_match', 24.0)

                # Get current state
                current_state = self._get_base_state()
                state = self._augment_state(
                    current_state, confidence, edge, odds,
                    time_to_match, implied_prob
                )

                # Get action from policy
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob, value = self.actor_critic.get_action(state_tensor)

                # Apply edge threshold
                if edge < self.min_edge_threshold:
                    action = 0.0

                # Execute in environment
                reward, done, info = self.env.step(
                    action, confidence, edge, odds,
                    self.bankroll, self.current_drawdown
                )

                # Update bankroll
                self.bankroll = info['new_bankroll']
                self.total_profit = self.bankroll - self.initial_bankroll

                # Update drawdown
                if self.bankroll > self.peak_bankroll:
                    self.peak_bankroll = self.bankroll
                self.current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll

                # Track bet history
                if action > 0:
                    self.total_bets += 1
                    if info['won']:
                        self.winning_bets += 1
                        self.consecutive_wins += 1
                        self.consecutive_losses = 0
                    else:
                        self.consecutive_losses += 1
                        self.consecutive_wins = 0

                    self.bet_history.append(info['profit'])
                    if len(self.bet_history) > 1000:
                        self.bet_history.pop(0)

                # Get next state
                next_state = self._augment_state(
                    self._get_base_state(), confidence, edge, odds,
                    time_to_match - 1, implied_prob
                )

                # Store experience
                experience = Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    log_prob=log_prob,
                    value=value
                )
                self.buffer.add(experience)

                episode_reward += reward
                episode_steps += 1

                if done:
                    break

            # Update policy at end of episode
            self._update_policy(next_value=0.0)

            episode_rewards.append(episode_reward)
            sharpe = self._calculate_sharpe_ratio()
            episode_sharpes.append(sharpe)

            # Track best policy by Sharpe ratio
            if sharpe > self.best_sharpe and episode > 10:
                self.best_sharpe = sharpe
                self.best_policy_state = {
                    k: v.cpu().clone() for k, v in self.actor_critic.state_dict().items()
                }

            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                avg_sharpe = np.mean(episode_sharpes[-10:])
                logger.info(f"Episode {episode+1}/{episodes}, "
                           f"Avg Reward: {avg_reward:.4f}, "
                           f"Avg Sharpe: {avg_sharpe:.3f}, "
                           f"Bankroll: {self.bankroll:.2f}, "
                           f"Drawdown: {self.current_drawdown:.2%}")

        # Load best policy
        if self.best_policy_state:
            self.actor_critic.load_state_dict(self.best_policy_state)

        self.trained_episodes = episodes
        self._update_model_hash()

        # Final metrics
        win_rate = self.winning_bets / max(1, self.total_bets)
        roi = self.total_profit / self.initial_bankroll
        max_dd = self._calculate_max_drawdown()

        logger.info(f"Training complete. Win Rate: {win_rate:.2%}, "
                   f"ROI: {roi:.2%}, Sharpe: {sharpe:.3f}, "
                   f"Max DD: {max_dd:.2%}")

        return {
            "model_type": self.model_type,
            "version": self.version,
            "model_hash": self.model_hash,
            "episodes_trained": self.trained_episodes,
            "avg_episode_reward": float(np.mean(episode_rewards)),
            "win_rate": float(win_rate),
            "roi": float(roi),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "total_bets": self.total_bets,
            "total_profit": float(self.total_profit),
            "final_bankroll": float(self.bankroll),
            "best_sharpe": float(self.best_sharpe)
        }

    def _get_base_state(self) -> np.ndarray:
        """Get base state vector (without model-specific features)."""
        state = np.zeros(RLConfig.state_dim)

        state[0] = self.bankroll / self.initial_bankroll
        state[1] = min(self.consecutive_losses, 10) / 10
        state[2] = min(self.consecutive_wins, 10) / 10

        if self.total_bets > 0:
            recent = self.bet_history[-min(20, len(self.bet_history)):]
            win_rate = sum(1 for p in recent if p > 0) / max(1, len(recent))
            state[3] = win_rate
        else:
            state[3] = 0.5

        if self.bet_history:
            recent = self.bet_history[-min(20, len(self.bet_history)):]
            avg_profit = np.mean(recent) / self.initial_bankroll
            state[4] = np.clip(avg_profit + 0.5, 0, 1)
        else:
            state[4] = 0.5

        if self.initial_bankroll > 0:
            roi = self.total_profit / self.initial_bankroll
            state[5] = np.clip(roi + 0.5, 0, 1)
        else:
            state[5] = 0.5

        return state

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate optimal stake recommendation."""
        confidence = features.get('confidence', {}).get('1x2', 0.6)
        edge = features.get('edge_vs_market', {}).get('best_edge_percent', 0) / 100
        odds = features.get('market_odds', {}).get(
            features.get('recommended_outcome', 'home'), 2.0
        )
        implied_prob = 1 / odds if odds > 0 else 0.5
        time_to_match = features.get('hours_until_match', 24.0)

        # Get current state
        current_state = self._get_base_state()
        state = self._augment_state(
            current_state, confidence, edge, odds,
            time_to_match, implied_prob
        )

        # Get deterministic action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, log_prob, value = self.actor_critic.get_action(
                state_tensor, deterministic=True
            )

        stake_percentage = action

        # Apply edge threshold
        if edge < self.min_edge_threshold:
            stake_percentage = 0.0

        # Apply drawdown protection
        if self.current_drawdown > self.drawdown_limit:
            stake_percentage *= (1 - self.current_drawdown)

        # Store for tracking
        self.last_state = state
        self.last_action = stake_percentage

        # Calculate Kelly for reference
        kelly = self._calculate_kelly(confidence, odds)

        return {
            "recommended_stake_percentage": float(stake_percentage),
            "recommended_stake_amount": float(stake_percentage * self.bankroll),
            "kelly_reference": float(kelly),
            "action_confidence": float(value),
            "current_bankroll": float(self.bankroll),
            "current_drawdown": float(self.current_drawdown),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "total_bets": self.total_bets,
            "win_rate": self.winning_bets / max(1, self.total_bets),
            "sharpe_ratio": self._calculate_sharpe_ratio(),
            "volatility": self._calculate_volatility(),
            "rl_policy_active": True,
            "edge_threshold_applied": edge >= self.min_edge_threshold,
            "drawdown_protection_active": self.current_drawdown > self.drawdown_limit,
            "model_hash": self.model_hash
        }

    def update_bet_result(
        self,
        stake_percentage: float,
        odds: float,
        won: bool,
        actual_profit: float
    ):
        """Update agent state after real bet settles."""
        # Update bankroll
        self.bankroll += actual_profit
        self.total_profit = self.bankroll - self.initial_bankroll

        # Update streaks
        if won:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.winning_bets += 1
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        self.total_bets += 1

        # Update drawdown
        if self.bankroll > self.peak_bankroll:
            self.peak_bankroll = self.bankroll
        self.current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll

        # Store bet history
        self.bet_history.append(actual_profit)
        if len(self.bet_history) > 1000:
            self.bet_history.pop(0)

        logger.info(f"Bet result: {'WIN' if won else 'LOSS'} | "
                   f"Stake: {stake_percentage:.2%} | "
                   f"Profit: {actual_profit:.2f} | "
                   f"Bankroll: {self.bankroll:.2f} | "
                   f"Drawdown: {self.current_drawdown:.2%} | "
                   f"Sharpe: {self._calculate_sharpe_ratio():.3f}")

    def get_confidence_score(self, market: str = "1x2") -> float:
        """Return confidence based on Sharpe ratio and win rate."""
        if self.total_bets < 20:
            return 0.5

        sharpe = self._calculate_sharpe_ratio()
        win_rate = self.winning_bets / self.total_bets

        # Confidence = normalized combination of Sharpe and win rate
        confidence = 0.5 + (sharpe * 0.3) + ((win_rate - 0.5) * 0.2)
        return float(np.clip(confidence, 0.3, 0.9))

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        sharpe = self._calculate_sharpe_ratio()
        max_dd = self._calculate_max_drawdown()
        win_rate = self.winning_bets / max(1, self.total_bets)
        roi = self.total_profit / self.initial_bankroll

        return {
            "total_bets": self.total_bets,
            "winning_bets": self.winning_bets,
            "win_rate": float(win_rate),
            "total_profit": float(self.total_profit),
            "roi": float(roi),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(max_dd),
            "current_bankroll": float(self.bankroll),
            "peak_bankroll": float(self.peak_bankroll),
            "current_drawdown": float(self.current_drawdown),
            "consecutive_losses": self.consecutive_losses,
            "consecutive_wins": self.consecutive_wins,
            "model_hash": self.model_hash,
            "trained_episodes": self.trained_episodes,
            "max_stake": self.max_stake,
            "min_edge_threshold": self.min_edge_threshold,
            "drawdown_limit": self.drawdown_limit
        }

    def save(self, path: str) -> None:
        """Save model to disk."""
        save_data = {
            'model_id': self.model_id,
            'model_name': self.model_name,
            'model_type': self.model_type,
            'version': self.version,
            'weight': self.weight,
            'params': self.params,
            'status': self.status,
            'model_hash': self.model_hash,
            'actor_critic_state': self.actor_critic.state_dict(),
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'epochs_per_update': self.epochs_per_update,
            'batch_size': self.batch_size,
            'max_stake': self.max_stake,
            'min_edge_threshold': self.min_edge_threshold,
            'drawdown_limit': self.drawdown_limit,
            'no_bet_penalty': self.no_bet_penalty,
            'risk_penalty_coef': self.risk_penalty_coef,
            'trained_episodes': self.trained_episodes,
            'best_sharpe': self.best_sharpe,
            'bankroll': self.bankroll,
            'initial_bankroll': self.initial_bankroll,
            'peak_bankroll': self.peak_bankroll,
            'total_bets': self.total_bets,
            'winning_bets': self.winning_bets,
            'total_profit': self.total_profit,
            'bet_history': self.bet_history[-1000:],
            'session_accuracies': {k.value: v for k, v in self.session_accuracies.items()},
            'final_score': self.final_score,
            'certified': self.certified
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"RL Policy Agent V{self.version} saved to {path}")

    def load(self, path: str) -> None:
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)

        self.model_id = data['model_id']
        self.model_name = data['model_name']
        self.model_type = data['model_type']
        self.version = data.get('version', 2)
        self.weight = data['weight']
        self.params = data['params']
        self.status = data['status']
        self.model_hash = data.get('model_hash', '')

        # Restore parameters
        self.learning_rate = data['learning_rate']
        self.gamma = data['gamma']
        self.gae_lambda = data['gae_lambda']
        self.clip_epsilon = data['clip_epsilon']
        self.epochs_per_update = data['epochs_per_update']
        self.batch_size = data['batch_size']
        self.max_stake = data['max_stake']
        self.min_edge_threshold = data['min_edge_threshold']
        self.drawdown_limit = data['drawdown_limit']
        self.no_bet_penalty = data.get('no_bet_penalty', 0.01)
        self.risk_penalty_coef = data.get('risk_penalty_coef', 0.1)
        self.trained_episodes = data.get('trained_episodes', 0)
        self.best_sharpe = data.get('best_sharpe', -float('inf'))

        # Restore bankroll (single source of truth)
        self.bankroll = data.get('bankroll', 1000.0)
        self.initial_bankroll = data.get('initial_bankroll', 1000.0)
        self.peak_bankroll = data.get('peak_bankroll', self.bankroll)
        self.total_bets = data.get('total_bets', 0)
        self.winning_bets = data.get('winning_bets', 0)
        self.total_profit = data.get('total_profit', 0.0)
        self.bet_history = data.get('bet_history', [])

        # Recompute derived state
        self.current_drawdown = (self.peak_bankroll - self.bankroll) / self.peak_bankroll if self.peak_bankroll > 0 else 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0

        # Rebuild network
        self.device = torch.device(RLConfig.device)
        self.actor_critic = ContinuousActorCritic(
            RLConfig.state_dim, RLConfig.hidden_dim, self.max_stake
        ).to(self.device)
        self.actor_critic.load_state_dict(data['actor_critic_state'])
        self.actor_critic.eval()

        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)
        self.buffer = RolloutBuffer()
        self.env = SimulatedBettingEnvironment(initial_bankroll=self.initial_bankroll)

        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)

        logger.info(f"RL Policy Agent V{self.version} loaded from {path}")