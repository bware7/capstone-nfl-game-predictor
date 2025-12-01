"""
NFL Game Predictor - Data Preparation
FIXED: Proper lagging of player stats to prevent data leakage
"""

import os
import sys

import nfl_data_py as nfl
import numpy as np
import pandas as pd

from config import config


class DataPreparer:
    """Prepare NFL data for modeling with proper temporal handling"""

    def __init__(self, seasons=None):
        self.seasons = seasons or config.seasons
        self.games_df = None
        self.player_stats = None

    def collect_game_data(self):
        """Collect game data from NFL API"""
        print(f"Collecting game data for seasons: {self.seasons}")

        games = nfl.import_schedules(self.seasons)

        # Filter to completed regular season games
        games = games[games['game_type'] == 'REG']
        games = games.dropna(subset=['home_score', 'away_score'])

        # Create target variable
        games['home_won'] = (games['home_score'] > games['away_score']).astype(int)

        # Sort chronologically - critical for time series
        games = games.sort_values(['season', 'week']).reset_index(drop=True)

        print(f"  Collected {len(games)} games")
        self.games_df = games
        return games

    def collect_player_stats(self):
        """Collect weekly player statistics"""
        print("Collecting player statistics...")

        try:
            weekly = nfl.import_weekly_data(self.seasons)

            # Safe pre-game stats only (no EPA - it's post-game)
            stat_cols = [
                'player_id',
                'player_name',
                'recent_team',
                'season',
                'week',
                'position',
                'completions',
                'attempts',
                'passing_yards',
                'passing_tds',
                'interceptions',
                'carries',
                'rushing_yards',
                'rushing_tds',
                'targets',
                'receptions',
                'receiving_yards',
                'receiving_tds',
                'fantasy_points_ppr',
            ]

            available_cols = [col for col in stat_cols if col in weekly.columns]
            weekly = weekly[available_cols]

            # Sort chronologically
            weekly = weekly.sort_values(['season', 'week']).reset_index(drop=True)

            print(f"  Collected stats for {len(weekly)} player-weeks")
            self.player_stats = weekly
            return weekly

        except Exception as e:
            print(f"  ERROR: Could not collect player stats: {e}")
            raise  # Fail fast - don't train on incomplete data

    def _compute_lagged_team_stats(self):
        """
        CRITICAL FIX: Compute rolling averages of player stats ENTERING each week.
        We can only use stats from PREVIOUS weeks, not the current week.
        """
        print("Computing lagged team statistics (preventing data leakage)...")

        if self.player_stats is None:
            print("  No player stats available")
            return

        # Stats to aggregate at team level
        agg_stats = [
            'completions',
            'attempts',
            'passing_yards',
            'passing_tds',
            'interceptions',
            'carries',
            'rushing_yards',
            'rushing_tds',
            'targets',
            'receptions',
            'receiving_yards',
            'receiving_tds',
            'fantasy_points_ppr',
        ]
        available_stats = [s for s in agg_stats if s in self.player_stats.columns]

        if not available_stats:
            print("  No stats available for aggregation")
            return

        # Aggregate player stats by team/week
        team_weekly = (
            self.player_stats.groupby(['recent_team', 'season', 'week'])[available_stats]
            .sum()
            .reset_index()
        )

        # Sort by team and time
        team_weekly = team_weekly.sort_values(['recent_team', 'season', 'week'])

        # CRITICAL: Compute rolling averages and SHIFT by 1 week
        # This ensures we only use stats from BEFORE the game
        rolling_cols = {}
        for stat in available_stats:
            col_name = f'{stat}_avg'
            # Rolling mean of last N games, shifted forward 1 week
            team_weekly[col_name] = team_weekly.groupby('recent_team')[stat].transform(
                lambda x: x.rolling(config.rolling_window, min_periods=1)
                .mean()
                .shift(config.lag_periods)
            )
            rolling_cols[stat] = col_name

        # Keep only the lagged averages (not raw stats which would leak)
        keep_cols = ['recent_team', 'season', 'week'] + list(rolling_cols.values())
        team_stats_lagged = team_weekly[keep_cols].copy()

        # Merge with games - home team
        self.games_df = self.games_df.merge(
            team_stats_lagged,
            left_on=['home_team', 'season', 'week'],
            right_on=['recent_team', 'season', 'week'],
            how='left',
            suffixes=('', '_drop'),
        )
        # Rename columns for home team
        for stat in rolling_cols.values():
            if stat in self.games_df.columns:
                self.games_df.rename(columns={stat: f'{stat}_home'}, inplace=True)

        # Drop merge key
        if 'recent_team' in self.games_df.columns:
            self.games_df.drop(columns=['recent_team'], inplace=True)

        # Merge with games - away team
        self.games_df = self.games_df.merge(
            team_stats_lagged,
            left_on=['away_team', 'season', 'week'],
            right_on=['recent_team', 'season', 'week'],
            how='left',
            suffixes=('', '_away'),
        )
        # Rename columns for away team
        for stat in rolling_cols.values():
            if stat in self.games_df.columns:
                if not stat.endswith('_home'):
                    self.games_df.rename(columns={stat: f'{stat}_away'}, inplace=True)

        # Drop merge key
        if 'recent_team' in self.games_df.columns:
            self.games_df.drop(columns=['recent_team'], inplace=True)

        # Create differentials
        for stat in rolling_cols.values():
            home_col = f'{stat}_home'
            away_col = f'{stat}_away'
            if home_col in self.games_df.columns and away_col in self.games_df.columns:
                self.games_df[f'{stat}_diff'] = self.games_df[home_col].fillna(0) - self.games_df[
                    away_col
                ].fillna(0)

        print(f"  Created {len(rolling_cols)} lagged team statistics")

    def engineer_team_rolling_stats(self):
        """Calculate rolling team performance (wins, points) - properly lagged"""
        print("Engineering team rolling statistics...")

        if self.games_df is None:
            return

        df = self.games_df.sort_values(['season', 'week']).copy()

        # We need to track each team's historical performance
        # This is complex because teams appear as both home and away

        # Create long format: one row per team-game
        home_games = df[
            ['season', 'week', 'home_team', 'home_score', 'away_score', 'home_won']
        ].copy()
        home_games.columns = ['season', 'week', 'team', 'points_for', 'points_against', 'won']

        away_games = df[
            ['season', 'week', 'away_team', 'away_score', 'home_score', 'home_won']
        ].copy()
        away_games.columns = ['season', 'week', 'team', 'points_for', 'points_against', 'won']
        away_games['won'] = 1 - away_games['won']  # Flip for away perspective

        all_games = pd.concat([home_games, away_games]).sort_values(['team', 'season', 'week'])

        # Calculate lagged rolling stats per team
        for stat in ['points_for', 'points_against', 'won']:
            all_games[f'{stat}_avg'] = all_games.groupby('team')[stat].transform(
                lambda x: x.rolling(config.rolling_window, min_periods=1).mean().shift(1)
            )

        # Pivot back to get home and away stats
        team_stats = all_games[
            ['season', 'week', 'team', 'points_for_avg', 'points_against_avg', 'won_avg']
        ]

        # Merge home team stats
        df = df.merge(
            team_stats,
            left_on=['season', 'week', 'home_team'],
            right_on=['season', 'week', 'team'],
            how='left',
        )
        df.rename(
            columns={
                'points_for_avg': 'home_pts_avg',
                'points_against_avg': 'home_pts_allowed_avg',
                'won_avg': 'home_win_pct',
            },
            inplace=True,
        )
        df.drop(columns=['team'], inplace=True, errors='ignore')

        # Merge away team stats
        df = df.merge(
            team_stats,
            left_on=['season', 'week', 'away_team'],
            right_on=['season', 'week', 'team'],
            how='left',
        )
        df.rename(
            columns={
                'points_for_avg': 'away_pts_avg',
                'points_against_avg': 'away_pts_allowed_avg',
                'won_avg': 'away_win_pct',
            },
            inplace=True,
        )
        df.drop(columns=['team'], inplace=True, errors='ignore')

        # Create differentials
        df['pts_avg_diff'] = df['home_pts_avg'].fillna(0) - df['away_pts_avg'].fillna(0)
        df['win_pct_diff'] = df['home_win_pct'].fillna(0) - df['away_win_pct'].fillna(0)

        self.games_df = df
        print("  Added rolling team performance stats")

    def add_betting_features(self):
        """Add betting market features if available"""
        print("Processing betting features...")

        df = self.games_df
        betting_cols = ['spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        available = [col for col in betting_cols if col in df.columns]

        if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
            # Convert moneyline to implied probability
            def ml_to_prob(ml):
                if pd.isna(ml):
                    return 0.5
                if ml < 0:
                    return abs(ml) / (abs(ml) + 100)
                elif ml > 0:
                    return 100 / (ml + 100)
                return 0.5

            df['home_implied_prob'] = df['home_moneyline'].apply(ml_to_prob)
            df['away_implied_prob'] = df['away_moneyline'].apply(ml_to_prob)
            print(f"  Added implied probability features")

        if 'spread_line' in df.columns:
            # Spread is from home team perspective (negative = home favored)
            df['spread_line'] = df['spread_line'].fillna(0)
            print(f"  Using spread line feature")

        self.games_df = df
        print(f"  Found {len(available)} betting columns")

    def add_contextual_features(self):
        """Add game context features"""
        print("Adding contextual features...")

        df = self.games_df

        # Rest advantage
        if 'home_rest' in df.columns and 'away_rest' in df.columns:
            df['rest_differential'] = df['home_rest'] - df['away_rest']
            print("  Added rest differential")

        # Division game
        if 'div_game' in df.columns:
            df['is_division'] = df['div_game'].astype(float)
            print("  Added division game indicator")

        # Season progress (week 1-18)
        if 'week' in df.columns:
            df['season_progress'] = df['week'] / 18.0
            print("  Added season progress")

        # Home field advantage (constant)
        df['home_field'] = 1

        self.games_df = df

    def validate_no_leakage(self):
        """Check for potential data leakage"""
        print("\nValidating for data leakage...")

        if 'home_won' not in self.games_df.columns:
            print("  No target variable found")
            return True

        y = self.games_df['home_won']
        leaky_features = []

        for col in self.games_df.select_dtypes(include=[np.number]).columns:
            if col == 'home_won':
                continue
            try:
                corr = abs(self.games_df[col].fillna(0).corr(y))
                if corr >= config.correlation_threshold:
                    leaky_features.append((col, corr))
            except:
                pass

        if leaky_features:
            print(f"  ⚠️ WARNING: Found {len(leaky_features)} potentially leaky features:")
            for feat, corr in sorted(leaky_features, key=lambda x: -x[1]):
                print(f"    - {feat}: {corr:.3f}")
            return False
        else:
            print("  ✅ No obvious leakage detected")
            return True

    def save_processed_data(self):
        """Save processed data to disk"""
        os.makedirs('data', exist_ok=True)

        output_file = config.game_data_path
        self.games_df.to_csv(output_file, index=False)
        print(f"  Saved games to {output_file}")

        if self.player_stats is not None:
            player_file = config.player_stats_path
            self.player_stats.to_csv(player_file, index=False)
            print(f"  Saved player stats to {player_file}")

    def run_full_pipeline(self, force_rebuild=False):
        """Run complete data preparation pipeline"""
        print("=" * 50)
        print("DATA PREPARATION PIPELINE")
        print("=" * 50)

        # Check if data exists
        if not force_rebuild and os.path.exists(config.game_data_path):
            print(f"\nData already exists at {config.game_data_path}")
            print("Use force_rebuild=True to regenerate")
            self.games_df = pd.read_csv(config.game_data_path)
            return self.games_df

        # Collect data
        self.collect_game_data()
        self.collect_player_stats()

        # Engineer features (with proper lagging)
        self._compute_lagged_team_stats()
        self.engineer_team_rolling_stats()
        self.add_betting_features()
        self.add_contextual_features()

        # Validate no leakage
        self.validate_no_leakage()

        # Save
        self.save_processed_data()

        print("\n✅ Data preparation complete!")
        print(f"   Games: {len(self.games_df)}")
        print(f"   Features: {len(self.games_df.columns)}")

        return self.games_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild data')
    args = parser.parse_args()

    preparer = DataPreparer()
    preparer.run_full_pipeline(force_rebuild=args.rebuild)
