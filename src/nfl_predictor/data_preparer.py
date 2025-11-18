"""
NFL Game Predictor - Data Preparation
Handles data collection and feature engineering
"""

import nfl_data_py as nfl
import numpy as np
import pandas as pd


class DataPreparer:
    """Prepare NFL data for modeling"""

    def __init__(self, seasons=[2023, 2024]):
        self.seasons = seasons

    def collect_game_data(self):
        """Collect game data from NFL API"""
        print(f"Collecting game data for seasons: {self.seasons}")

        # Get game data
        games = nfl.import_schedules(self.seasons)

        # Filter to completed games
        games = games[games['game_type'] == 'REG']
        games = games.dropna(subset=['home_score', 'away_score'])

        # Create target variable
        games['home_won'] = (games['home_score'] > games['away_score']).astype(int)

        print(f"  Collected {len(games)} games")
        return games

    def collect_player_stats(self):
        """Collect weekly player statistics"""
        print("Collecting player statistics...")

        try:
            # Get weekly player stats
            weekly = nfl.import_weekly_data(self.seasons)

            # Filter to relevant columns
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

            print(f"  Collected stats for {len(weekly)} player-weeks")
            return weekly
        except Exception as e:
            print(f"  Could not collect player stats: {e}")
            return None

    def add_betting_data(self, games_df):
        """Add betting lines and odds to game data"""
        print("Adding betting market data...")

        # This would typically come from an odds API
        # For now, using what's in the data if available
        betting_cols = ['spread_line', 'total_line', 'home_moneyline', 'away_moneyline']
        available = [col for col in betting_cols if col in games_df.columns]

        if available:
            print(f"  Found {len(available)} betting columns")

        return games_df

    def engineer_team_averages(self, games_df, window=3):
        """Calculate rolling team averages"""
        print(f"Calculating {window}-game rolling averages...")

        # Sort by team and week
        games_df = games_df.sort_values(['season', 'week'])

        # Features to average
        metrics = ['score', 'yards', 'turnovers'] if 'yards' in games_df.columns else ['score']

        for metric in metrics:
            if f'home_{metric}' in games_df.columns:
                # Calculate rolling averages
                for team_type in ['home', 'away']:
                    col_name = f'{team_type}_{metric}'
                    if col_name in games_df.columns:
                        # Group by team and calculate rolling mean
                        # This is simplified - in production would track by actual team
                        games_df[f'{col_name}_avg_{window}'] = games_df.groupby(
                            f'{team_type}_team'
                        )[col_name].transform(
                            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
                        )

        print(f"  Added rolling averages for {len(metrics)} metrics")
        return games_df

    def prepare_final_dataset(self, games_df, player_stats_df=None):
        """Combine all data sources into final dataset"""
        print("\nPreparing final dataset...")

        # Remove incomplete games
        games_df = games_df.dropna(subset=['home_won'])

        # Add basic features
        games_df['home_field_advantage'] = 1

        # Add rest days if available
        if 'home_rest' in games_df.columns and 'away_rest' in games_df.columns:
            games_df['rest_advantage'] = games_df['home_rest'] - games_df['away_rest']

        # Save processed data
        output_file = 'data/games_with_features.csv'
        games_df.to_csv(output_file, index=False)
        print(f"  Saved to {output_file}")

        if player_stats_df is not None:
            player_file = 'data/weekly_stats.csv'
            player_stats_df.to_csv(player_file, index=False)
            print(f"  Saved player stats to {player_file}")

        return games_df

    def run_full_pipeline(self):
        """Run complete data preparation pipeline"""
        print("=" * 50)
        print("DATA PREPARATION PIPELINE")
        print("=" * 50)

        # Collect data
        games = self.collect_game_data()
        player_stats = self.collect_player_stats()

        # Add features
        games = self.add_betting_data(games)
        games = self.engineer_team_averages(games)

        # Prepare final dataset
        final_data = self.prepare_final_dataset(games, player_stats)

        print("\n✅ Data preparation complete!")
        print(f"   Games: {len(final_data)}")
        print(f"   Features: {len(final_data.columns)}")

        return final_data


if __name__ == "__main__":
    preparer = DataPreparer(seasons=[2023, 2024])
    preparer.run_full_pipeline()
