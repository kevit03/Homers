# NBAGPT

A GPT-style transformer trained from scratch on NBA play-by-play. Each game is a sequence of event tokens (`H_3PT_MAKE`, `A_DREB`, `H_TOV`, ...), and the model learns to predict the next event and the home team's win probability at every moment of the game.

## Dashboard

`export_dashboard.py` runs the model once and bakes every prediction into a single self-contained `dashboard.html` (about 2 MB). No server is needed: open it in a browser, email it, or put it on any static host (GitHub Pages, Netlify, Vercel, S3).

```bash
python export_dashboard.py --ckpt runs/base/best.pt --data data/processed/games.pkl
open dashboard.html
```

On a Mac, double-click **`Open Dashboard.command`**. It rebuilds from the real model if one exists, otherwise from the synthetic demo.

The page has:

- **Scoreboard**: Brier score vs. the score-and-clock baseline, next-play accuracy, and perplexity on the test set.
- **Game replay**: sort games by most dramatic, closest, or biggest upset. Hover or scrub (arrow keys work) to see the win probability, the model's top five guesses for the next play, the biggest momentum swings, and the full play-by-play, which you can copy as CSV.
- **Model quality**: error by quarter, calibration, and training history.
- **What-if**: a quarter, clock, and margin calculator using the baseline.

Re-run the export after each training run to refresh it.

## Player profiles & fantasy

```bash
python fetch_rosters.py     # names, numbers, positions, heights for the profile cards (~2 min)
```

Nothing else is needed: `export_dashboard.py` builds profiles from `data/raw` (via `export_profiles.py`, cached in `data/processed/profiles_cache.pkl`) whenever the data is real.

- **Clickable players**: names in the replay (last play, momentum swings, play-by-play) and an **On the floor** strip open a player card with his headshot (from cdn.nba.com), bio, season averages, shooting vs. league average, a game-by-game fantasy chart, recent games, and his career in the dataset. `#player-<personId>` in the URL opens a card directly.
- **Live context**: open a card while replaying a game he's in and it shows whether he's on the floor, his box score so far, and his odds of making the next play. NBAGPT predicts the team's next play; his share of each play type is his season rate (blended with 60 league-average minutes) against the other four players on the floor. During a free-throw trip, the remaining free throws go to the shooter. It uses the model's top five guesses, so it slightly understates his total involvement.
- **Fantasy leaderboard**: FP/game with NBA.com scoring (PTS ×1, REB ×1.2, AST ×1.5, STL ×3, BLK ×3, TOV −1), top-5 cards, filters and sortable columns.
- Minutes and lineups are rebuilt in game-clock order and match official minutes to within about 0.1 mpg for the stars checked (Jokić, Giannis, Embiid, LeBron in 2021-22). Complete lineups are known for about 97% of plays. Headshots use the player's current photo, so it may show a newer team's jersey.

## Player, defender & coach model

```bash
python fetch_context.py     # head coaches, Synergy play types, per-game defensive matchups (~4 h, resumable)
./refresh_players.sh        # build shots, train the shot model, re-export the dashboard
```

- **`build_shots.py`** writes one row per shot. Each row has the shot type (layup, dunk, floater, hook, short/long mid-range, corner 3, above-the-break 3), the style (pull-up, step-back, driving, cutting, putback, …), all 10 players on the floor (rebuilt from substitutions), the likely primary defender (from the NBA matchup feed), both head coaches, and the game situation.
- **`shot_model.py`** predicts shot type, style and make probability. It starts from each player's own history and learns how the defender, the lineups, each coach and each team's play-type mix shift it. It's scored against league-average and player-history baselines.
- The dashboard gains **Player tendencies** (shot chart, shot diet, Synergy play types, a matchup simulator that runs the model in the browser), **Defender indicators**, and **Coaches & schemes**.

Limits: the defender is the opponent who guarded the shooter most in that game, not a per-shot assignment. The coach is the team's listed head coach for the season, so mid-season changes aren't tracked.

## Pipeline

```bash
pip install -r requirements.txt

python fetch_data.py --seasons 2021-22 2022-23 2023-24 2024-25   # ~5k games, a few hours, resumable
python tokenize_pbp.py                                           # -> data/processed/games.pkl
python baseline.py                                               # logistic regression on score + time
python train.py --out runs/base                                  # 4-layer, 128-dim model (~0.8M params)
python evaluate.py --ckpt runs/base/best.pt                      # metrics + plots in results/
python scaling.py                                                # model size vs. val loss
```

Test the full pipeline without the NBA API first:

```bash
python synthetic.py --games 400
python tokenize_pbp.py --raw data/raw_synthetic --out data/processed/synthetic.pkl
python baseline.py --data data/processed/synthetic.pkl
python train.py --data data/processed/synthetic.pkl --out runs/syn --n_layer 2 --n_embd 64 --n_head 2 --lr 1e-3
python evaluate.py --ckpt runs/syn/best.pt --data data/processed/synthetic.pkl --out results_syn
```

## Design

- **Tokens**: 31-token vocabulary. Events are team-relative (home/away), and offensive vs. defensive rebounds are inferred from who missed last. Substitutions and replays are dropped.
- **Game state**: time remaining, score differential, period, and lead-relative-to-time are projected into the embedding at each position, so the model doesn't have to reconstruct the score by counting baskets.
- **Model**: decoder-only transformer with causal attention, tied embeddings, and two heads: next event (cross-entropy) and home win (BCE at every position).
- **Split**: the latest season is the test set, so there's no leakage from future games.

## What to report

- Next-event perplexity and accuracy
- Win-probability Brier score vs. the logistic baseline, overall and by quarter
- Calibration plot
- Scaling curve (val loss vs. parameters)
- Win-probability charts for memorable games

## Extensions

- Player tokens or learned player embeddings (then cluster them)
- Lineup-aware win probability, possession-level expected points
- Game simulation by sampling future events
- Train on 10+ seasons and extend the scaling curve
