# app/services/team_mapper.py
import re
import logging
from typing import Dict, List, Optional, Tuple
from difflib import get_close_matches
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.models import Team

logger = logging.getLogger(__name__)


class TeamMapper:
    """
    Team mapping service for handling external API team IDs.

    Features:
        - Persistent mapping storage
        - Fuzzy name matching
        - Manual override support
    """

    # Common team name variations
    TEAM_ALIASES = {
        "Manchester United": ["Man United", "Man Utd", "MUFC", "Manchester Utd"],
        "Manchester City": ["Man City", "MCFC", "Manchester C"],
        "Liverpool": ["LFC", "Liverpool FC"],
        "Chelsea": ["CFC", "Chelsea FC"],
        "Arsenal": ["AFC", "Arsenal FC", "The Gunners"],
        "Tottenham": ["Spurs", "Tottenham Hotspur", "THFC"],
        "Newcastle": ["NUFC", "Newcastle United"],
        "Aston Villa": ["Villa", "AVFC"],
        "West Ham": ["WHUFC", "West Ham United"],
        "Everton": ["EFC", "Everton FC"],
    }

    def __init__(self, db: AsyncSession):
        self.db = db
        self._cache: Dict[str, str] = {}

    async def get_internal_id(self, external_id: str, external_name: str) -> Optional[str]:
        """Get internal team ID from external source"""
        # Check cache
        if external_id in self._cache:
            return self._cache[external_id]

        # Query database
        result = await self.db.execute(
            select(Team).where(Team.external_id == external_id)
        )
        team = result.scalar_one_or_none()

        if team:
            self._cache[external_id] = team.id
            return team.id

        # Try fuzzy matching by name
        team = await self._find_by_name(external_name)

        if team:
            # Store mapping
            team.external_id = external_id
            await self.db.commit()
            self._cache[external_id] = team.id
            return team.id

        logger.warning(f"Could not map team: {external_name} (ID: {external_id})")
        return None

    async def _find_by_name(self, name: str) -> Optional[Team]:
        """Find team by name with fuzzy matching"""
        # Direct match
        result = await self.db.execute(
            select(Team).where(Team.name == name)
        )
        team = result.scalar_one_or_none()

        if team:
            return team

        # Check aliases
        for canonical, aliases in self.TEAM_ALIASES.items():
            if name in aliases or canonical == name:
                result = await self.db.execute(
                    select(Team).where(Team.name == canonical)
                )
                return result.scalar_one_or_none()

        # Fuzzy match
        result = await self.db.execute(select(Team.name))
        all_names = [row[0] for row in result.all()]

        matches = get_close_matches(name, all_names, n=1, cutoff=0.6)

        if matches:
            result = await self.db.execute(
                select(Team).where(Team.name == matches[0])
            )
            return result.scalar_one_or_none()

        return None

    async def create_team(self, external_id: str, name: str, **kwargs) -> Team:
        """Create a new team entry"""
        team = Team(
            external_id=external_id,
            name=name,
            **kwargs
        )
        self.db.add(team)
        await self.db.commit()
        await self.db.refresh(team)

        self._cache[external_id] = team.id
        return team