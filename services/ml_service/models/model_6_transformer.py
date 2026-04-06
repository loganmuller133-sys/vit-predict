# services/ml-service/models/model_6_transformer.py
import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
import logging
import pickle
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from datetime import datetime
from dataclasses import dataclass
import math
from sklearn.preprocessing import StandardScaler

_NNBase = nn.Module if TORCH_AVAILABLE else object

from app.models.base_model import BaseModel, MarketType, Session

logger = logging.getLogger(__name__)


@dataclass
class TransformerConfig:
    """Configuration for Transformer model."""
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_seq_len: int = 20
    num_features: int = 32
    num_classes_1x2: int = 3
    num_classes_ou: int = 2
    num_classes_btts: int = 2


class PositionalEncoding(_NNBase):
    """Sinusoidal positional encoding with dropout."""

    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TimeAwareAttention(_NNBase):
    """Attention with time decay injected BEFORE softmax."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)

        # Learnable time decay parameter
        self.time_weight = nn.Parameter(torch.ones(1) * 0.1)

        # Standard multihead attention
        self.attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

    def forward(self, query, key, value, time_deltas=None, key_padding_mask=None):
        """
        Time-aware attention with decay applied BEFORE softmax.
        """
        # Compute standard attention scores
        d_k = self.d_model // self.nhead
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # Apply time decay BEFORE softmax (correct place)
        if time_deltas is not None:
            # time_deltas shape: (batch, seq_len)
            # Create time decay matrix
            time_decay = torch.exp(-self.time_weight * time_deltas.unsqueeze(-1))
            # Apply decay to scores
            scores = scores * time_decay

        # Apply mask if provided
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float('-inf'))

        # Softmax (now with time decay applied)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Compute output
        output = torch.matmul(attn_weights, value)

        return output, attn_weights


class TransformerModel(_NNBase):
    """Transformer with time-aware attention and multi-head outputs."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.input_embedding = nn.Linear(config.num_features, config.d_model)

        # Layer norm for input stability
        self.input_norm = nn.LayerNorm(config.d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(config.d_model, config.max_seq_len, config.dropout)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Time-aware attention layer
        self.time_attention = TimeAwareAttention(config.d_model, config.nhead, config.dropout)

        # Output heads
        self.head_1x2 = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, config.num_classes_1x2)
        )

        self.head_ou = nn.Sequential(
            nn.Linear(config.d_model, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.num_classes_ou)
        )

        self.head_btts = nn.Sequential(
            nn.Linear(config.d_model, 32),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(32, config.num_classes_btts)
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, time_deltas=None, mask=None):
        # Embed and normalize
        x = self.input_embedding(x)
        x = self.input_norm(x)
        x = self.pos_encoder(x)

        # Apply transformer encoder
        if mask is not None:
            padding_mask = mask == 0
            x = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        else:
            x = self.transformer_encoder(x)

        # Apply time-aware attention
        x, attn_weights = self.time_attention(x, x, x, time_deltas, mask)

        # Masked mean pooling
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / (mask_expanded.sum(dim=1) + 1e-8)
        else:
            x = x.mean(dim=1)

        # Output heads
        logits_1x2 = self.head_1x2(x)
        logits_ou = self.head_ou(x)
        logits_btts = self.head_btts(x)

        return logits_1x2, logits_ou, logits_btts, attn_weights


if TORCH_AVAILABLE:
    class MatchSequenceDataset(Dataset):
        """Dataset with proper temporal isolation."""

        def __init__(self, sequences, time_deltas, targets_1x2, targets_ou, targets_btts, masks=None):
            self.sequences = torch.FloatTensor(sequences)
            self.time_deltas = torch.FloatTensor(time_deltas)
            self.targets_1x2 = torch.LongTensor(targets_1x2)
            self.targets_ou = torch.LongTensor(targets_ou)
            self.targets_btts = torch.LongTensor(targets_btts)
            self.masks = torch.FloatTensor(masks) if masks is not None else None

        def __len__(self):
            return len(self.targets_1x2)

        def __getitem__(self, idx):
            if self.masks is not None:
                return (self.sequences[idx], self.time_deltas[idx], self.masks[idx],
                        self.targets_1x2[idx], self.targets_ou[idx], self.targets_btts[idx])
            return (self.sequences[idx], self.time_deltas[idx],
                    self.targets_1x2[idx], self.targets_ou[idx], self.targets_btts[idx])
else:
    MatchSequenceDataset = None


class TransformerSequenceModel(BaseModel):
    """
    Transformer Sequence Model V2 - Fixed and production-ready.

    Fixes applied:
        - Real features (not placeholder constants)
        - Proper feature scaling with StandardScaler
        - Log-scaled time deltas
        - Time decay applied BEFORE softmax
        - Softmax in prediction phase
        - Strict temporal cutoff (no leakage)
        - Class weights for imbalance
        - Complete training loop with early stopping
    """

    def __init__(
        self,
        model_name: str,
        weight: float = 1.0,
        version: int = 2,
        params: Optional[Dict[str, Any]] = None,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 15,
        learning_rate: float = 0.0001,
        batch_size: int = 32,
        epochs: int = 100,
        patience: int = 10,
        device: Optional[str] = None
    ):
        super().__init__(
            model_name=model_name,
            model_type="Transformer",
            weight=weight,
            version=version,
            params=params,
            supported_markets=[
                MarketType.MATCH_ODDS,
                MarketType.OVER_UNDER,
                MarketType.BTTS
            ]
        )

        # Model configuration
        self.config = TransformerConfig(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_features=20,  # Will be set during training
            num_classes_1x2=3,
            num_classes_ou=2,
            num_classes_btts=2
        )

        # Training parameters
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        if device is None:
            self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Model components
        self.model: Optional[TransformerModel] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.ReduceLROnPlateau] = None

        # Feature scaling
        self.feature_scaler: Optional[StandardScaler] = None
        self.num_features: int = 20

        # Class weights for imbalance
        self.class_weights_1x2: Optional[torch.Tensor] = None
        self.class_weights_ou: Optional[torch.Tensor] = None
        self.class_weights_btts: Optional[torch.Tensor] = None

        # Training state
        self.trained_matches_count: int = 0
        self.best_val_loss: float = float('inf')
        self.best_model_state: Optional[Dict] = None

        # Attention weights for explainability
        self.last_attention_weights: Optional[np.ndarray] = None

        # Only certify if PyTorch is available
        self.certified = TORCH_AVAILABLE

    def _extract_real_features(
        self,
        match: Dict[str, Any],
        team_name: str,
        team_history: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Extract REAL features (no placeholders).

        Features (20 total):
            0. Goals scored (last 5 avg)
            1. Goals conceded (last 5 avg)
            2. Points per game (last 5)
            3. xG (rolling avg)
            4. xG conceded (rolling avg)
            5. Shots on target avg
            6. Possession avg
            7. Home/away indicator
            8. Days since last match (log scaled)
            9. Opponent strength (Elo/rating)
            10. Form trend (last 3 vs previous 3)
            11. Goal difference rolling
            12. Clean sheet rate (last 5)
            13. Scored rate (last 5)
            14. Rest days advantage
            15. Home streak (consecutive home matches)
            16. Win streak (consecutive wins)
            17. Undefeated streak
            18. xG differential
            19. Shot accuracy
        """
        features = []

        # Calculate rolling stats from history
        last_5 = team_history[-5:] if len(team_history) >= 5 else team_history
        last_3 = team_history[-3:] if len(team_history) >= 3 else team_history
        prev_3 = team_history[-6:-3] if len(team_history) >= 6 else []

        # Goals scored/conceded (last 5 avg)
        goals_for = [m.get('goals_for', 0) for m in last_5]
        goals_against = [m.get('goals_against', 0) for m in last_5]
        features.append(np.mean(goals_for) if goals_for else 1.5)
        features.append(np.mean(goals_against) if goals_against else 1.5)

        # Points per game (last 5)
        points = [m.get('points', 1) for m in last_5]
        features.append(np.mean(points) / 3 if points else 0.5)

        # xG and xG conceded
        xg = [m.get('xg', 1.2) for m in last_5]
        xg_conceded = [m.get('xg_conceded', 1.2) for m in last_5]
        features.append(np.mean(xg) / 3 if xg else 0.4)
        features.append(np.mean(xg_conceded) / 3 if xg_conceded else 0.4)

        # Shots on target avg
        shots = [m.get('shots_on_target', 4) for m in last_5]
        features.append(min(np.mean(shots) / 10, 1.0) if shots else 0.4)

        # Possession avg
        possession = [m.get('possession', 50) for m in last_5]
        features.append(np.mean(possession) / 100 if possession else 0.5)

        # Home/away indicator
        is_home = match.get('is_home', True)
        features.append(1.0 if is_home else 0.0)

        # Days since last match (log scaled)
        days = match.get('days_since_last_match', 7)
        features.append(math.log1p(min(days, 30)) / math.log1p(30))

        # Opponent strength (Elo rating normalized)
        opponent_rating = match.get('opponent_rating', 1500)
        features.append((opponent_rating - 1200) / 800)  # 1200-2000 range to 0-1

        # Form trend (last 3 vs previous 3 points per game)
        last_3_ppg = np.mean([m.get('points', 1) for m in last_3]) / 3 if last_3 else 0.5
        prev_3_ppg = np.mean([m.get('points', 1) for m in prev_3]) / 3 if prev_3 else 0.5
        features.append(max(-0.5, min(0.5, last_3_ppg - prev_3_ppg)) + 0.5)

        # Goal difference rolling
        gd = [m.get('goals_for', 0) - m.get('goals_against', 0) for m in last_5]
        gd_norm = (np.mean(gd) + 3) / 6 if gd else 0.5  # -3 to +3 -> 0-1
        features.append(gd_norm)

        # Clean sheet rate (last 5)
        clean_sheets = [1 if m.get('goals_against', 1) == 0 else 0 for m in last_5]
        features.append(np.mean(clean_sheets) if clean_sheets else 0.2)

        # Scored rate (last 5)
        scored = [1 if m.get('goals_for', 0) > 0 else 0 for m in last_5]
        features.append(np.mean(scored) if scored else 0.6)

        # Rest days advantage
        rest_adv = match.get('rest_days_advantage', 0)
        features.append(max(0, min(1, (rest_adv + 7) / 14)))

        # Home streak
        home_streak = match.get('home_streak', 0)
        features.append(min(home_streak / 10, 1.0))

        # Win streak
        win_streak = match.get('win_streak', 0)
        features.append(min(win_streak / 5, 1.0))

        # Undefeated streak
        undefeated = match.get('undefeated_streak', 0)
        features.append(min(undefeated / 10, 1.0))

        # xG differential
        xg_diff = match.get('xg_differential', 0)
        features.append(max(0, min(1, (xg_diff + 2) / 4)))

        # Shot accuracy
        shot_acc = match.get('shot_accuracy', 0.4)
        features.append(min(shot_acc, 1.0))

        return features[:20]  # Ensure exactly 20 features

    def _extract_sequence_features(
        self,
        team_matches: List[Dict[str, Any]],
        team_name: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract feature sequence with proper temporal ordering.
        No leakage - features only from matches before current.
        """
        features = []
        time_deltas = []
        prev_date = None

        for i, match in enumerate(team_matches):
            # Use only matches before current for feature extraction
            history = team_matches[:i]

            match_features = self._extract_real_features(
                {**match, 'is_home': match.get('home_team') == team_name},
                team_name,
                history
            )
            features.append(match_features)

            # Calculate time delta (log scaled)
            match_date = match.get('match_date')
            if match_date and prev_date:
                delta = (match_date - prev_date).days
                log_delta = math.log1p(min(delta, 60)) / math.log1p(60)
                time_deltas.append(log_delta)
            else:
                time_deltas.append(0)
            prev_date = match_date

        # Pad or truncate
        seq_len = len(features)
        if seq_len > self.config.max_seq_len:
            features = features[-self.config.max_seq_len:]
            time_deltas = time_deltas[-self.config.max_seq_len:]
            seq_len = self.config.max_seq_len

        # Pad if shorter
        if seq_len < self.config.max_seq_len:
            pad_len = self.config.max_seq_len - seq_len
            features = [[0.0] * self.num_features] * pad_len + features
            time_deltas = [0.0] * pad_len + time_deltas
            mask = [0] * pad_len + [1] * seq_len
        else:
            mask = [1] * seq_len

        return np.array(features), np.array(time_deltas), np.array(mask)

    def _build_sequences_strict(
        self,
        matches: List[Dict[str, Any]],
        is_training: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Build sequences with STRICT temporal cutoff.
        Each match appears in training OR validation, not both.
        """
        # Sort matches chronologically
        matches_sorted = sorted(matches, key=lambda x: x.get('match_date', datetime.min))

        # Group by team, maintaining temporal order
        team_sequences: Dict[str, List[Dict]] = defaultdict(list)

        for match in matches_sorted:
            # Add to home team
            team_sequences[match.get('home_team')].append({
                **match,
                'team': match.get('home_team'),
                'is_home': True,
                'goals_for': match.get('home_goals', 0),
                'goals_against': match.get('away_goals', 0),
                'points': 3 if match.get('home_goals', 0) > match.get('away_goals', 0)
                         else (1 if match.get('home_goals', 0) == match.get('away_goals', 0) else 0)
            })

            # Add to away team
            team_sequences[match.get('away_team')].append({
                **match,
                'team': match.get('away_team'),
                'is_home': False,
                'goals_for': match.get('away_goals', 0),
                'goals_against': match.get('home_goals', 0),
                'points': 3 if match.get('away_goals', 0) > match.get('home_goals', 0)
                         else (1 if match.get('away_goals', 0) == match.get('home_goals', 0) else 0)
            })

        sequences = []
        time_deltas_list = []
        masks_list = []
        targets_1x2_list = []
        targets_ou_list = []
        targets_btts_list = []

        # Build sequences per team
        for team, history in team_sequences.items():
            if len(history) < 3:
                continue

            for i in range(len(history) - 1):
                # Sequence: matches up to i
                seq_matches = history[:i+1]
                target_match = history[i+1]

                # Extract features
                features, t_deltas, mask = self._extract_sequence_features(seq_matches, team)

                # Target outcomes
                target_1x2 = target_match.get('points', 1)  # 3/1/0 but need 2/1/0
                if target_match.get('points', 1) == 3:
                    target_1x2 = 2
                elif target_match.get('points', 1) == 1:
                    target_1x2 = 1
                else:
                    target_1x2 = 0

                total_goals = (target_match.get('home_goals', 0) + 
                              target_match.get('away_goals', 0))
                target_ou = 1 if total_goals > 2.5 else 0

                target_btts = 1 if (target_match.get('home_goals', 0) > 0 and 
                                    target_match.get('away_goals', 0) > 0) else 0

                sequences.append(features)
                time_deltas_list.append(t_deltas)
                masks_list.append(mask)
                targets_1x2_list.append(target_1x2)
                targets_ou_list.append(target_ou)
                targets_btts_list.append(target_btts)

        if not sequences:
            return (np.array([]), np.array([]), np.array([]), 
                    np.array([]), np.array([]), np.array([]))

        X = np.array(sequences, dtype=np.float32)
        time_deltas = np.array(time_deltas_list, dtype=np.float32)
        masks = np.array(masks_list, dtype=np.float32)
        y_1x2 = np.array(targets_1x2_list)
        y_ou = np.array(targets_ou_list)
        y_btts = np.array(targets_btts_list)

        # Scale features if training
        if is_training:
            X_flat = X.reshape(-1, X.shape[-1])
            self.feature_scaler = StandardScaler()
            X_scaled_flat = self.feature_scaler.fit_transform(X_flat)
            X = X_scaled_flat.reshape(X.shape)
            self.num_features = X.shape[-1]
            self.config.num_features = self.num_features

            # Calculate class weights for imbalance
            self._calculate_class_weights(y_1x2, y_ou, y_btts)

        elif self.feature_scaler is not None:
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled_flat = self.feature_scaler.transform(X_flat)
            X = X_scaled_flat.reshape(X.shape)

        return X, time_deltas, masks, y_1x2, y_ou, y_btts

    def _calculate_class_weights(self, y_1x2, y_ou, y_btts):
        """Calculate class weights for handling imbalance."""
        # 1X2 weights
        unique, counts = np.unique(y_1x2, return_counts=True)
        total = len(y_1x2)
        weights_1x2 = torch.FloatTensor([total / (len(unique) * counts[i]) for i in range(len(unique))])
        self.class_weights_1x2 = weights_1x2.to(self.device)

        # Over/Under weights
        unique_ou, counts_ou = np.unique(y_ou, return_counts=True)
        weights_ou = torch.FloatTensor([total / (2 * counts_ou[i]) for i in range(len(unique_ou))])
        self.class_weights_ou = weights_ou.to(self.device)

        # BTTS weights
        unique_btts, counts_btts = np.unique(y_btts, return_counts=True)
        weights_btts = torch.FloatTensor([total / (2 * counts_btts[i]) for i in range(len(unique_btts))])
        self.class_weights_btts = weights_btts.to(self.device)

        logger.info(f"Class weights - 1X2: {self.class_weights_1x2.tolist()}")

    def train(
        self,
        matches: List[Dict[str, Any]],
        validation_split: float = 0.2
    ) -> Dict[str, Any]:
        """Train transformer model with proper temporal split."""
        if not matches:
            return {"error": "No training data"}

        # Sort by date
        matches_sorted = sorted(matches, key=lambda x: x.get('match_date', datetime.min))

        # Time-based split (no leakage)
        split_idx = int(len(matches_sorted) * (1 - validation_split))
        train_matches = matches_sorted[:split_idx]
        val_matches = matches_sorted[split_idx:]

        logger.info(f"Training on {len(train_matches)} matches, validating on {len(val_matches)}")

        # Build sequences
        logger.info("Building training sequences...")
        X_train, t_train, m_train, y1_train, y2_train, y3_train = self._build_sequences_strict(train_matches, True)

        logger.info("Building validation sequences...")
        X_val, t_val, m_val, y1_val, y2_val, y3_val = self._build_sequences_strict(val_matches, False)

        if X_train.shape[0] == 0:
            return {"error": "Not enough sequences for training"}

        logger.info(f"Training sequences: {X_train.shape}, Validation sequences: {X_val.shape}")

        # Set device
        device = self.device
        logger.info(f"Using device: {device}")

        # Initialize model
        self.model = TransformerModel(self.config).to(device)

        # Optimizer with weight decay
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )

        # Loss functions with class weights
        criterion_1x2 = nn.CrossEntropyLoss(weight=self.class_weights_1x2)
        criterion_ou = nn.CrossEntropyLoss(weight=self.class_weights_ou)
        criterion_btts = nn.CrossEntropyLoss(weight=self.class_weights_btts)

        # Create dataloaders
        train_dataset = MatchSequenceDataset(X_train, t_train, y1_train, y2_train, y3_train, m_train)
        val_dataset = MatchSequenceDataset(X_val, t_val, y1_val, y2_val, y3_val, m_val)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if len(batch) == 6:
                    X, t, m, y1, y2, y3 = batch
                else:
                    X, t, y1, y2, y3 = batch
                    m = None

                X = X.to(device)
                t = t.to(device)
                y1 = y1.to(device)
                y2 = y2.to(device)
                y3 = y3.to(device)
                if m is not None:
                    m = m.to(device)

                self.optimizer.zero_grad()

                logits_1x2, logits_ou, logits_btts, _ = self.model(X, t, m)

                loss1 = criterion_1x2(logits_1x2, y1)
                loss2 = criterion_ou(logits_ou, y2)
                loss3 = criterion_btts(logits_btts, y3)
                loss = loss1 + loss2 + loss3

                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(logits_1x2, 1)
                train_total += y1.size(0)
                train_correct += (predicted == y1).sum().item()

            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch) == 6:
                        X, t, m, y1, y2, y3 = batch
                    else:
                        X, t, y1, y2, y3 = batch
                        m = None

                    X = X.to(device)
                    t = t.to(device)
                    y1 = y1.to(device)
                    if m is not None:
                        m = m.to(device)

                    logits_1x2, logits_ou, logits_btts, _ = self.model(X, t, m)

                    loss1 = criterion_1x2(logits_1x2, y1)
                    val_loss += loss1.item()

                    _, predicted = torch.max(logits_1x2, 1)
                    val_total += y1.size(0)
                    val_correct += (predicted == y1).sum().item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total

            self.scheduler.step(avg_val_loss)

            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                self.best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                self.best_val_loss = best_val_loss
            else:
                patience_counter += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{self.epochs}: "
                           f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                           f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")

            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)

        self.model.to(device)
        self.trained_matches_count = len(train_matches)

        logger.info(f"Training complete. Best validation loss: {best_val_loss:.4f}, Best accuracy: {val_acc:.4f}")

        return {
            "model_type": self.model_type,
            "version": self.version,
            "matches_trained": self.trained_matches_count,
            "validation_loss": best_val_loss,
            "validation_accuracy": val_acc,
            "num_sequences": X_train.shape[0],
            "num_features": self.num_features,
            "max_seq_len": self.config.max_seq_len,
            "device": device
        }

    async def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Generate predictions with softmax and proper scaling."""
        if not self.model:
            return self._fallback_prediction()

        home_history = features.get('home_team_history', [])
        away_history = features.get('away_team_history', [])

        if not home_history or not away_history:
            return self._fallback_prediction()

        # Build sequences
        home_features, home_time, home_mask = self._extract_sequence_features(home_history, features.get('home_team'))
        away_features, away_time, away_mask = self._extract_sequence_features(away_history, features.get('away_team'))

        # Use home sequence as primary
        X = np.expand_dims(home_features, axis=0)
        t = np.expand_dims(home_time, axis=0)
        m = np.expand_dims(home_mask, axis=0)

        # Scale using trained scaler
        if self.feature_scaler is not None:
            X_flat = X.reshape(-1, X.shape[-1])
            X_scaled_flat = self.feature_scaler.transform(X_flat)
            X = X_scaled_flat.reshape(X.shape)

        # Convert to tensors
        device = self.device
        X_tensor = torch.FloatTensor(X).to(device)
        t_tensor = torch.FloatTensor(t).to(device)
        m_tensor = torch.FloatTensor(m).to(device)

        # Predict with softmax
        self.model.eval()
        with torch.no_grad():
            logits_1x2, logits_ou, logits_btts, attn_weights = self.model(X_tensor, t_tensor, m_tensor)

            # Apply softmax for probabilities
            probs_1x2 = F.softmax(logits_1x2, dim=1).cpu().numpy()[0]
            probs_ou = F.softmax(logits_ou, dim=1).cpu().numpy()[0]
            probs_btts = F.softmax(logits_btts, dim=1).cpu().numpy()[0]

            self.last_attention_weights = attn_weights.cpu().numpy()

        # Map outputs (0=away, 1=draw, 2=home)
        home_prob = float(probs_1x2[2])
        draw_prob = float(probs_1x2[1])
        away_prob = float(probs_1x2[0])

        # Normalize
        total = home_prob + draw_prob + away_prob
        if total > 0:
            home_prob /= total
            draw_prob /= total
            away_prob /= total

        over_25_prob = float(probs_ou[1])
        under_25_prob = float(probs_ou[0])

        btts_prob = float(probs_btts[1])
        no_btts_prob = float(probs_btts[0])

        # Calculate confidence from attention entropy
        if self.last_attention_weights is not None:
            attention_entropy = -np.sum(self.last_attention_weights * 
                                        np.log(self.last_attention_weights + 1e-8))
            max_entropy = np.log(self.config.max_seq_len)
            confidence = 1 - (attention_entropy / max_entropy)
        else:
            confidence = 0.6

        return {
            "home_prob": home_prob,
            "draw_prob": draw_prob,
            "away_prob": away_prob,
            "over_2_5_prob": over_25_prob,
            "under_2_5_prob": under_25_prob,
            "btts_prob": btts_prob,
            "no_btts_prob": no_btts_prob,
            "home_goals_expectation": home_prob * 2.5,
            "away_goals_expectation": away_prob * 2.2,
            "confidence": {
                "1x2": float(confidence),
                "over_under": 0.6,
                "btts": 0.6
            },
            "attention_pattern": "distributed" if confidence < 0.5 else "focused",
            "sequence_used": True,
            "sequence_length": len(home_history)
        }

    def _fallback_prediction(self) -> Dict[str, Any]:
        """Return fallback prediction."""
        return {
            "home_prob": 0.34,
            "draw_prob": 0.33,
            "away_prob": 0.33,
            "over_2_5_prob": 0.5,
            "under_2_5_prob": 0.5,
            "btts_prob": 0.5,
            "no_btts_prob": 0.5,
            "home_goals_expectation": 1.5,
            "away_goals_expectation": 1.2,
            "confidence": {"1x2": 0.5, "over_under": 0.5, "btts": 0.5},
            "attention_pattern": "none",
            "sequence_used": False,
            "sequence_length": 0
        }

    def get_confidence_score(self, market: str = "1x2") -> float:
        """Return confidence based on attention entropy."""
        if self.last_attention_weights is not None:
            attention_entropy = -np.sum(self.last_attention_weights * 
                                        np.log(self.last_attention_weights + 1e-8))
            max_entropy = np.log(self.config.max_seq_len)
            return float(1 - (attention_entropy / max_entropy))
        return 0.6

    def get_attention_weights(self) -> Optional[np.ndarray]:
        """Get attention weights for explainability."""
        return self.last_attention_weights

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
            'model_state': self.model.state_dict() if self.model else None,
            'feature_scaler': self.feature_scaler,
            'config': self.config,
            'num_features': self.num_features,
            'trained_matches_count': self.trained_matches_count,
            'best_val_loss': self.best_val_loss,
            'session_accuracies': {k.value: v for k, v in self.session_accuracies.items()},
            'final_score': self.final_score,
            'certified': self.certified
        }

        with open(path, 'wb') as f:
            pickle.dump(save_data, f)

        logger.info(f"Transformer model V{self.version} saved to {path}")

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
        self.feature_scaler = data['feature_scaler']
        self.config = data['config']
        self.num_features = data['num_features']
        self.trained_matches_count = data['trained_matches_count']
        self.best_val_loss = data.get('best_val_loss', float('inf'))

        # Rebuild model
        if data['model_state']:
            device = self.device
            self.model = TransformerModel(self.config).to(device)
            self.model.load_state_dict(data['model_state'])
            self.model.eval()

        # Restore certification data
        for session_val, accuracy in data.get('session_accuracies', {}).items():
            self.session_accuracies[Session(session_val)] = accuracy
        self.final_score = data.get('final_score')
        self.certified = data.get('certified', False)

        logger.info(f"Transformer model V{self.version} loaded from {path}")