#!/usr/bin/env python
# scripts/generate_historical_data.py
"""Generate synthetic historical football match data for training"""

import json
import random
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HistoricalDataGenerator:
    """Generate synthetic historical match data"""
    
    TEAMS = {
        "premier_league": [
            "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
            "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
            "Leicester", "Liverpool", "Manchester City", "Manchester United",
            "Newcastle", "Nottingham", "Southampton", "Tottenham", "West Ham", "Wolverhampton"
        ],
        "la_liga": [
            "Real Madrid", "Barcelona", "Atletico Madrid", "Sevilla", "Valencia",
            "Villarreal", "Real Sociedad", "Girona", "Betis", "Athletic Bilbao"
        ],
        "bundesliga": [
            "Bayern Munich", "Borussia Dortmund", "Leverkusen", "RB Leipzig",
            "Frankfurt", "Union Berlin", "Freiburg", "Hoffenheim", "Wolfsburg", "Cologne"
        ]
    }
    
    def __init__(self):
        self.matches = []
    
    def generate(self, league: str = "premier_league", num_matches: int = 500):
        """Generate synthetic match data"""
        logger.info(f"Generating {num_matches} historical matches for {league}...")
        teams = self.TEAMS.get(league, self.TEAMS["premier_league"])
        
        # Generate matches
        base_date = datetime.now() - timedelta(days=365)
        
        for i in range(num_matches):
            # Random date in the past year
            match_date = base_date + timedelta(days=random.randint(0, 365))
            
            # Random teams
            home_team = random.choice(teams)
            away_team = random.choice([t for t in teams if t != home_team])
            
            # Generate realistic goals (Poisson-like distribution)
            home_goals = self._generate_goals()
            away_goals = self._generate_goals()
            
            # Determine outcome
            if home_goals > away_goals:
                outcome = "home"
            elif away_goals > home_goals:
                outcome = "away"
            else:
                outcome = "draw"
            
            match = {
                "home_team": home_team,
                "away_team": away_team,
                "league": league,
                "match_date": match_date.isoformat(),
                "home_goals": home_goals,
                "away_goals": away_goals,
                "outcome": outcome,
                "total_goals": home_goals + away_goals,
                "home_corners": random.randint(3, 12),
                "away_corners": random.randint(3, 12),
                "home_shots": random.randint(5, 20),
                "away_shots": random.randint(5, 20),
                "home_xg": round(random.uniform(0.5, 3.5), 2),
                "away_xg": round(random.uniform(0.5, 3.5), 2),
                "btts": 1 if home_goals > 0 and away_goals > 0 else 0
            }
            
            self.matches.append(match)
        
        logger.info(f"Generated {len(self.matches)} matches")
        return self.matches
    
    def _generate_goals(self):
        """Generate realistic goal distribution"""
        rand = random.random()
        if rand < 0.125:
            return 0
        elif rand < 0.310:
            return 1
        elif rand < 0.485:
            return 2
        elif rand < 0.615:
            return 3
        else:
            return random.randint(4, 6)
    
    def save(self, filepath: str):
        """Save matches to JSON file"""
        logger.info(f"Saving {len(self.matches)} matches to {filepath}")
        with open(filepath, 'w') as f:
            json.dump(self.matches, f, indent=2)
        logger.info(f"Saved to {filepath}")
    
    def distribute_by_season(self, matches_per_season: int = 380):
        """Distribute matches by season"""
        seasons = {}
        for match in self.matches:
            date = datetime.fromisoformat(match["match_date"])
            season = f"{date.year}-{date.year+1}"
            if season not in seasons:
                seasons[season] = []
            seasons[season].append(match)
        
        logger.info(f"Distributed matches into {len(seasons)} seasons")
        return seasons


def main():
    """Generate and save historical data"""
    gen = HistoricalDataGenerator()
    
    # Generate for multiple leagues
    for league, num_matches in [
        ("premier_league", 800),
        ("la_liga", 500),
        ("bundesliga", 500)
    ]:
        gen.generate(league, num_matches)
    
    # Save to file
    gen.save("/workspaces/vit-predict/data/historical_matches.json")
    
    logger.info(f"Total matches generated: {len(gen.matches)}")
    logger.info("Data generation complete!")


if __name__ == "__main__":
    main()