"""
extract_match_metrics.py
------------------------
Extracts 150 metrics from a single Opta JSON match file.
Outputs one row per team (home / away) with all metrics as columns.

Usage:
    python extract_match_metrics.py <path_to_json>
    python extract_match_metrics.py          # uses built-in demo file

Qualifier reference (Opta):
  Q1   = headed flag         Q2  = big chance flag
  Q9   = penalty flag        Q13 = jersey number
  Q20  = blocked flag        Q55 = sequence ID
  Q56  = body part (Left / Right / Center / Back)
  Q72  = head clearance flag Q102 = shot x
  Q103 = shot y              Q107 = corner flag
  Q140 = pass end_x          Q141 = pass end_y
  Q144 = pass sequence #     Q147 = ?
  Q155 = big chance flag     Q210 = assist shot type (15/16)
  Q212 = goal-mouth x        Q213 = goal-mouth y
  Q230 = shot x (near goal)  Q231 = shot y
  Q233 = time in seconds     Q279 = goal-post zone (G/S)
  Q285 = direct free kick    Q286 = corner assist flag
"""

import json
import math
import sys
import os
import glob
import numpy as np
import pandas as pd
from collections import defaultdict

# ── pitch zones ───────────────────────────────────────────────────────────────
def in_box(x, y):
    return x > 83 and 21 < y < 79

def in_6yd(x, y):
    return x > 94 and 30 < y < 70

def in_final_third(x):
    return x > 66

def in_mid_third(x):
    return 33 < x <= 66

def in_own_third(x):
    return x <= 33

def in_left_channel(y):
    return y < 33

def in_right_channel(y):
    return y > 67

def in_central(y):
    return 33 <= y <= 67

def pass_distance(x1, y1, x2, y2):
    # pitch is 105m × 68m scaled to 100×100
    dx = (x2 - x1) / 100 * 105
    dy = (y2 - y1) / 100 * 68
    return math.hypot(dx, dy)

def is_progressive(x1, x2):
    return (x2 - x1) >= 10 and x2 > 50

# ── qualifier helpers ─────────────────────────────────────────────────────────
def quals(event):
    return {q['qualifierId']: q.get('value') for q in event.get('qualifier', [])}

def qval(event, qid):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value')
    return None

def has_q(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

def sfloat(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default

# ── type ID sets ──────────────────────────────────────────────────────────────
SHOT_IDS     = {13, 14, 15, 16}
ON_TARGET    = {15, 16}
GOAL_IDS     = {16}
PASS_ID      = 1
DRIBBLE_ID   = 3
TACKLE_IDS   = {7, 51}
INTERC_ID    = 8
CLEAR_IDS    = {12, 40, 70}
AERIAL_IDS   = {44, 52}
FOUL_ID      = 4
CARD_ID      = 17
BLOCK_IDS    = {74, 50}
RECOVERY_IDS = {30, 49}
CORNER_ID    = 6
GOAL_KICK_ID = 41
SAVE_IDS     = {10, 28}
SUB_OFF      = 18
SUB_ON       = 19

# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(json_path: str) -> pd.DataFrame:
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)

    meta = d.get('matchDetails', {})
    events = d['event']
    fname  = os.path.basename(json_path)
    date_str = fname.split('_')[0] if '_' in fname else ''
    teams_str = fname.replace('.json','').split('_', 1)[1] if '_' in fname else fname
    parts = teams_str.split(' - ', 1)
    home_name = parts[0].strip() if len(parts) == 2 else 'Home'
    away_name = parts[1].strip() if len(parts) == 2 else 'Away'

    scores    = meta.get('scores', {})
    ft        = scores.get('ft', {})
    ht        = scores.get('ht', {})
    periods   = {p['id']: p for p in meta.get('period', [])}
    p1_len    = periods.get(1, {}).get('lengthMin', 45)
    p2_len    = periods.get(2, {}).get('lengthMin', 45)
    p1_inj    = periods.get(1, {}).get('announcedInjuryTime', 0)
    p2_inj    = periods.get(2, {}).get('announcedInjuryTime', 0)
    match_len = meta.get('matchLengthMin', 90)

    # identify home/away by contestant order in events
    home_cid = away_cid = None
    for e in events:
        cid = e.get('contestantId')
        if cid and home_cid is None:
            home_cid = cid
        elif cid and cid != home_cid and away_cid is None:
            away_cid = cid
        if home_cid and away_cid:
            break

    def side(cid):
        return 'home' if cid == home_cid else 'away'

    # ── split events by team ─────────────────────────────────────────────────
    team_events = {'home': [], 'away': []}
    for e in events:
        s = side(e.get('contestantId', ''))
        team_events[s].append(e)

    # ── compute per-team metrics dict ─────────────────────────────────────────
    def team_metrics(evts, side_key, opp_evts):
        m = {}

        # --- PASSING (metrics 1-25) ------------------------------------------
        passes    = [e for e in evts if e.get('typeId') == PASS_ID]
        succ_pass = [e for e in passes if e.get('outcome') == 1]
        fail_pass = [e for e in passes if e.get('outcome') != 1]

        m['passes_total']          = len(passes)
        m['passes_successful']     = len(succ_pass)
        m['passes_failed']         = len(fail_pass)
        m['pass_accuracy_pct']     = round(len(succ_pass) / max(len(passes), 1) * 100, 1)

        dists = []
        fwd = bwd = lat = 0
        into_ft = into_box = progressive = 0
        p_own = p_mid = p_final = 0
        p_left = p_right = p_central = 0
        long_p = short_p = med_p = 0
        crosses = 0
        pass_right = pass_left = pass_head = 0
        acc_own = acc_mid = acc_final = 0
        cnt_own = cnt_mid = cnt_final = 0

        for e in passes:
            x1, y1 = sfloat(e.get('x'), 50), sfloat(e.get('y'), 50)
            x2 = sfloat(qval(e, 140))
            y2 = sfloat(qval(e, 141))
            ok = e.get('outcome') == 1
            dist = pass_distance(x1, y1, x2, y2) if x2 and y2 else 0

            dists.append(dist)
            dx = x2 - x1 if x2 else 0
            if   dx >  3:  fwd += 1
            elif dx < -3:  bwd += 1
            else:          lat += 1

            if x2 and in_final_third(x2):   into_ft   += 1
            if x2 and y2 and in_box(x2, y2): into_box  += 1
            if x2 and is_progressive(x1, x2): progressive += 1

            if   in_own_third(x1):   p_own  += 1; cnt_own   += 1; acc_own   += int(ok)
            elif in_mid_third(x1):   p_mid  += 1; cnt_mid   += 1; acc_mid   += int(ok)
            else:                    p_final+= 1; cnt_final += 1; acc_final += int(ok)

            if   in_left_channel(y1):   p_left    += 1
            elif in_right_channel(y1):  p_right   += 1
            else:                       p_central += 1

            if   dist > 32:  long_p  += 1
            elif dist < 15:  short_p += 1
            else:            med_p   += 1

            bp = str(qval(e, 56) or '').lower()
            if   'right' in bp: pass_right += 1
            elif 'left'  in bp: pass_left  += 1
            elif 'back'  in bp: pass_head  += 1

            # cross: wide position, pass into final third
            if (y1 < 20 or y1 > 80) and x2 and in_final_third(x2):
                crosses += 1

        m['passes_forward']        = fwd
        m['passes_backward']       = bwd
        m['passes_lateral']        = lat
        m['passes_into_final_third'] = into_ft
        m['passes_into_box']       = into_box
        m['passes_progressive']    = progressive
        m['passes_own_third']      = p_own
        m['passes_mid_third']      = p_mid
        m['passes_final_third']    = p_final
        m['passes_left_channel']   = p_left
        m['passes_right_channel']  = p_right
        m['passes_central']        = p_central
        m['passes_long']           = long_p
        m['passes_medium']         = med_p
        m['passes_short']          = short_p
        m['crosses']               = crosses
        m['pass_avg_length_m']     = round(float(np.mean(dists)) if dists else 0, 2)
        m['pass_max_length_m']     = round(float(np.max(dists))  if dists else 0, 2)
        m['passes_right_foot']     = pass_right
        m['passes_left_foot']      = pass_left
        m['passes_headed']         = pass_head
        m['pass_acc_own_third']    = round(acc_own   / max(cnt_own,   1) * 100, 1)
        m['pass_acc_mid_third']    = round(acc_mid   / max(cnt_mid,   1) * 100, 1)
        m['pass_acc_final_third']  = round(acc_final / max(cnt_final, 1) * 100, 1)
        # key passes = passes whose next event is a shot
        m['key_passes']            = sum(1 for e in passes if has_q(e, 210))

        # --- SHOOTING (metrics 26-50) ----------------------------------------
        shots    = [e for e in evts if e.get('typeId') in SHOT_IDS]
        on_tgt   = [e for e in shots if e.get('typeId') in ON_TARGET]
        goals    = [e for e in shots if e.get('typeId') == 16]
        goals_p1 = [e for e in goals if e.get('periodId') == 1]
        goals_p2 = [e for e in goals if e.get('periodId') == 2]

        m['shots_total']           = len(shots)
        m['shots_on_target']       = len(on_tgt)
        m['shots_off_target']      = sum(1 for e in shots if e.get('typeId') == 13)
        m['shots_post']            = sum(1 for e in shots if e.get('typeId') == 14)
        m['goals']                 = len(goals)
        m['goals_first_half']      = len(goals_p1)
        m['goals_second_half']     = len(goals_p2)
        m['shot_accuracy_pct']     = round(len(on_tgt) / max(len(shots), 1) * 100, 1)
        m['conversion_rate_pct']   = round(len(goals) / max(len(shots), 1) * 100, 1)

        s_box = s_6yd = s_out = 0
        s_right = s_left = s_head = 0
        s_dists = []
        for e in shots:
            x, y = sfloat(e.get('x'), 50), sfloat(e.get('y'), 50)
            if in_6yd(x, y):         s_6yd += 1
            elif in_box(x, y):       s_box += 1
            else:                    s_out += 1
            bp = str(qval(e, 56) or '').lower()
            if   'right' in bp: s_right += 1
            elif 'left'  in bp: s_left  += 1
            elif 'back'  in bp: s_head  += 1
            # distance to goal: Q213 or calculate
            dval = sfloat(qval(e, 213), None)
            if dval is None:
                dval = pass_distance(x, y, 100, 50)
            s_dists.append(dval)

        m['shots_in_6yd_box']      = s_6yd
        m['shots_in_penalty_area'] = s_box
        m['shots_outside_box']     = s_out
        m['shots_right_foot']      = s_right
        m['shots_left_foot']       = s_left
        m['shots_headed']          = s_head
        m['shot_avg_distance_m']   = round(float(np.mean(s_dists)) if s_dists else 0, 2)
        m['big_chances']           = sum(1 for e in evts if has_q(e, 155))
        m['goals_from_inside_box'] = sum(1 for e in goals if in_box(sfloat(e.get('x')), sfloat(e.get('y'))))
        m['goals_outside_box']     = sum(1 for e in goals if not in_box(sfloat(e.get('x')), sfloat(e.get('y'))))
        m['goals_right_foot']      = sum(1 for e in goals if 'right' in str(qval(e,56) or '').lower())
        m['goals_left_foot']       = sum(1 for e in goals if 'left'  in str(qval(e,56) or '').lower())
        m['goals_headed']          = sum(1 for e in goals if 'back'  in str(qval(e,56) or '').lower())
        m['shots_blocked']         = sum(1 for e in evts if e.get('typeId') in BLOCK_IDS)
        m['penalties_awarded']     = sum(1 for e in evts if e.get('typeId') == 56)
        m['penalty_goals']         = sum(1 for e in goals if has_q(e, 9))

        # --- DEFENDING (metrics 51-70) ----------------------------------------
        tackles = [e for e in evts if e.get('typeId') in TACKLE_IDS]
        m['tackles_total']         = len(tackles)
        m['tackles_won']           = sum(1 for e in tackles if e.get('outcome') == 1)
        m['tackles_lost']          = sum(1 for e in tackles if e.get('outcome') != 1)
        m['tackle_success_pct']    = round(m['tackles_won'] / max(m['tackles_total'], 1) * 100, 1)
        m['interceptions']         = sum(1 for e in evts if e.get('typeId') == INTERC_ID)
        m['clearances']            = sum(1 for e in evts if e.get('typeId') in CLEAR_IDS)
        m['clearances_headed']     = sum(1 for e in evts if e.get('typeId') == 40)
        m['blocks']                = sum(1 for e in evts if e.get('typeId') in BLOCK_IDS)
        m['aerial_duels_total']    = sum(1 for e in evts if e.get('typeId') in AERIAL_IDS)
        m['aerial_duels_won']      = sum(1 for e in evts if e.get('typeId') in AERIAL_IDS and e.get('outcome') == 1)
        m['aerial_success_pct']    = round(m['aerial_duels_won'] / max(m['aerial_duels_total'], 1) * 100, 1)
        m['fouls_committed']       = sum(1 for e in evts if e.get('typeId') == FOUL_ID and e.get('outcome') == 0)
        m['fouls_won']             = sum(1 for e in evts if e.get('typeId') == FOUL_ID and e.get('outcome') == 1)
        cards = [e for e in evts if e.get('typeId') == CARD_ID]
        m['yellow_cards']          = sum(1 for e in cards if qval(e, 44) in ('Yellow', '31'))
        m['red_cards']             = sum(1 for e in cards if qval(e, 44) in ('Red', '32'))
        m['gk_saves']              = sum(1 for e in evts if e.get('typeId') in SAVE_IDS)
        # defensive actions in opponent half (high press indicator)
        m['def_actions_opp_half']  = sum(1 for e in evts
                                         if e.get('typeId') in (INTERC_ID, *TACKLE_IDS, *CLEAR_IDS, *BLOCK_IDS)
                                         and sfloat(e.get('x'), 0) > 50)
        m['errors_leading_shot']   = sum(1 for e in evts if e.get('typeId') == 59)
        m['errors_leading_goal']   = sum(1 for e in evts if e.get('typeId') == 61)
        m['offsides']              = sum(1 for e in evts if e.get('typeId') == 5 and has_q(e, 286))

        # --- POSSESSION / DRIBBLES / TEMPO (metrics 71-95) -------------------
        dribbles = [e for e in evts if e.get('typeId') == DRIBBLE_ID]
        m['dribbles_total']        = len(dribbles)
        m['dribbles_successful']   = sum(1 for e in dribbles if e.get('outcome') == 1)
        m['dribble_success_pct']   = round(m['dribbles_successful'] / max(m['dribbles_total'], 1) * 100, 1)
        m['ball_recoveries']       = sum(1 for e in evts if e.get('typeId') in RECOVERY_IDS)
        m['ball_touches']          = sum(1 for e in evts if e.get('typeId') == 43)
        m['total_actions']         = len(evts)

        # possession proxy: fraction of all ball events
        total_ball = len(evts) + len(opp_evts)
        m['possession_pct']        = round(len(evts) / max(total_ball, 1) * 100, 1)

        # territorial split
        m['actions_own_third']     = sum(1 for e in evts if in_own_third(sfloat(e.get('x'), 50)))
        m['actions_mid_third']     = sum(1 for e in evts if in_mid_third(sfloat(e.get('x'), 50)))
        m['actions_final_third']   = sum(1 for e in evts if in_final_third(sfloat(e.get('x'), 50)))
        m['actions_own_half']      = sum(1 for e in evts if sfloat(e.get('x'), 50) < 50)
        m['actions_opp_half']      = sum(1 for e in evts if sfloat(e.get('x'), 50) >= 50)
        m['opp_half_pct']          = round(m['actions_opp_half'] / max(m['total_actions'], 1) * 100, 1)

        # PPDA (passes per defensive action): opp passes / own defensive actions
        opp_passes = sum(1 for e in opp_evts if e.get('typeId') == PASS_ID)
        own_def    = sum(1 for e in evts if e.get('typeId') in (INTERC_ID, *TACKLE_IDS, FOUL_ID))
        m['ppda']                  = round(opp_passes / max(own_def, 1), 2)

        # actions per minute
        m['actions_per_minute']    = round(m['total_actions'] / max(match_len, 1), 2)

        # period split
        m['actions_period1']       = sum(1 for e in evts if e.get('periodId') == 1)
        m['actions_period2']       = sum(1 for e in evts if e.get('periodId') == 2)
        m['shots_period1']         = sum(1 for e in evts if e.get('typeId') in SHOT_IDS and e.get('periodId') == 1)
        m['shots_period2']         = sum(1 for e in evts if e.get('typeId') in SHOT_IDS and e.get('periodId') == 2)
        m['passes_period1']        = sum(1 for e in passes if e.get('periodId') == 1)
        m['passes_period2']        = sum(1 for e in passes if e.get('periodId') == 2)

        # --- SET PIECES (metrics 96-110) -------------------------------------
        m['corners_taken']         = sum(1 for e in evts if e.get('typeId') == CORNER_ID)
        m['goal_kicks']            = sum(1 for e in evts if e.get('typeId') == GOAL_KICK_ID)
        m['free_kicks']            = sum(1 for e in evts if e.get('typeId') == PASS_ID and has_q(e, 5))
        m['throw_ins']             = sum(1 for e in evts if e.get('typeId') == PASS_ID and has_q(e, 107))
        m['direct_free_kicks']     = sum(1 for e in evts if has_q(e, 285))
        m['corners_leading_shot']  = 0  # placeholder (chain analysis)
        m['set_piece_goals']       = sum(1 for e in goals if has_q(e, 9) or has_q(e, 286))
        m['gk_distribution']       = sum(1 for e in evts if e.get('typeId') in (27, 34, GOAL_KICK_ID))

        # --- SUBSTITUTIONS / SQUAD (metrics 111-120) -------------------------
        m['substitutions_made']    = sum(1 for e in evts if e.get('typeId') == SUB_OFF)
        m['unique_players']        = len(set(e.get('playerName') for e in evts if e.get('playerName')))

        # top contributor by actions
        player_acts = defaultdict(int)
        for e in evts:
            pn = e.get('playerName')
            if pn:
                player_acts[pn] += 1
        top_player = max(player_acts, key=player_acts.get) if player_acts else ''
        m['top_player_by_actions'] = top_player
        m['top_player_action_count'] = player_acts.get(top_player, 0)

        # top scorer
        scorer_goals = defaultdict(int)
        for e in goals:
            pn = e.get('playerName')
            if pn:
                scorer_goals[pn] += 1
        top_scorer = max(scorer_goals, key=scorer_goals.get) if scorer_goals else ''
        m['top_scorer']            = top_scorer
        m['top_scorer_goals']      = scorer_goals.get(top_scorer, 0)

        # most passes
        player_passes = defaultdict(int)
        for e in passes:
            pn = e.get('playerName')
            if pn:
                player_passes[pn] += 1
        top_passer = max(player_passes, key=player_passes.get) if player_passes else ''
        m['top_passer']            = top_passer
        m['top_passer_count']      = player_passes.get(top_passer, 0)

        # most shots
        player_shots = defaultdict(int)
        for e in shots:
            pn = e.get('playerName')
            if pn:
                player_shots[pn] += 1
        top_shooter = max(player_shots, key=player_shots.get) if player_shots else ''
        m['top_shooter']           = top_shooter
        m['top_shooter_shots']     = player_shots.get(top_shooter, 0)

        # --- MATCH RESULT FROM THIS TEAM'S POV (metrics 121-130) ------------
        m['result']                = 'win' if meta.get('winner') == side_key else \
                                     ('draw' if meta.get('winner') == 'draw' else 'loss')
        m['goals_scored']          = ft.get(side_key, 0)
        m['goals_conceded']        = ft.get('away' if side_key == 'home' else 'home', 0)
        m['ht_goals_scored']       = ht.get(side_key, 0)
        m['ht_goals_conceded']     = ht.get('away' if side_key == 'home' else 'home', 0)
        m['goal_difference']       = m['goals_scored'] - m['goals_conceded']
        m['clean_sheet']           = int(m['goals_conceded'] == 0)
        m['came_from_behind']      = int(m['ht_goals_scored'] < m['ht_goals_conceded'] and m['result'] == 'win')
        m['threw_away_lead']       = int(m['ht_goals_scored'] > m['ht_goals_conceded'] and m['result'] == 'loss')

        # --- TEMPO / INTENSITY (metrics 131-145) -----------------------------
        # minute of first shot
        first_shot_min = min((e.get('timeMin', 999) for e in shots), default=None)
        m['first_shot_minute']     = first_shot_min if first_shot_min is not None else -1
        first_goal_min = min((e.get('timeMin', 999) for e in goals), default=None)
        m['first_goal_minute']     = first_goal_min if first_goal_min is not None else -1

        # last 15 min shots (min 75+)
        m['shots_last_15min']      = sum(1 for e in shots if e.get('timeMin', 0) >= 75)
        m['goals_last_15min']      = sum(1 for e in goals if e.get('timeMin', 0) >= 75)
        m['shots_first_15min']     = sum(1 for e in shots if e.get('timeMin', 0) <= 15)

        # avg shot x (how deep into attack)
        shot_xs = [sfloat(e.get('x'), 50) for e in shots]
        m['avg_shot_x']            = round(float(np.mean(shot_xs)) if shot_xs else 0, 2)

        # avg pass start x
        pass_xs = [sfloat(e.get('x'), 50) for e in passes]
        m['avg_pass_start_x']      = round(float(np.mean(pass_xs)) if pass_xs else 0, 2)

        # shot quality proxy (on target / total) × goals
        m['shot_quality_index']    = round(m['shot_accuracy_pct'] * m['conversion_rate_pct'] / 100, 2)

        # pass-before-shot chain count
        m['sequences_ending_shot'] = sum(1 for e in shots if len(e.get('qualifier',[])) > 3)

        # high turnover count (ball lost in opp half)
        m['turnovers_opp_half']    = sum(1 for e in evts
                                         if e.get('typeId') == PASS_ID
                                         and e.get('outcome') != 1
                                         and sfloat(e.get('x'), 0) > 50)

        m['turnovers_own_half']    = sum(1 for e in evts
                                         if e.get('typeId') == PASS_ID
                                         and e.get('outcome') != 1
                                         and sfloat(e.get('x'), 0) <= 50)

        m['deep_completions']      = sum(1 for e in succ_pass
                                         if sfloat(qval(e,140) or 0) > 83)

        m['crosses_successful']    = sum(1 for e in passes
                                         if (sfloat(e.get('y'),50) < 20 or sfloat(e.get('y'),50) > 80)
                                         and sfloat(qval(e,140) or 0) > 66
                                         and e.get('outcome') == 1)

        m['carry_progression']     = sum(1 for e in evts if e.get('typeId') == 43
                                         and sfloat(e.get('x'), 50) > 50)

        # --- ADVANCED RATIOS (metrics 141-150) --------------------------------
        m['shots_per_pass']        = round(len(shots) / max(len(passes), 1), 4)
        m['progressive_pass_pct']  = round(progressive / max(len(passes), 1) * 100, 1)
        m['passes_into_box_pct']   = round(into_box   / max(len(passes), 1) * 100, 1)
        m['def_vs_opp_pass_ratio'] = round(own_def    / max(opp_passes, 1), 4)
        m['goals_per_shot_on_tgt'] = round(len(goals) / max(len(on_tgt), 1), 3)
        m['recovery_rate']         = round(m['ball_recoveries'] / max(m['total_actions'], 1) * 100, 2)
        m['attack_third_pass_pct'] = round(p_final / max(len(passes), 1) * 100, 1)
        m['pressure_index']        = round(m['def_actions_opp_half'] / max(m['total_actions'], 1) * 100, 2)
        m['shot_dominance']        = round(len(shots) / max(len(shots) + sum(1 for e in opp_evts if e.get('typeId') in SHOT_IDS), 1) * 100, 1)

        # --- MATCH META (metrics 150) ----------------------------------------
        m['match_length_min']      = match_len
        m['period1_length_min']    = p1_len
        m['period2_length_min']    = p2_len
        m['period1_injury_time_s'] = p1_inj
        m['period2_injury_time_s'] = p2_inj

        return m

    home_m = team_metrics(team_events['home'], 'home', team_events['away'])
    away_m = team_metrics(team_events['away'], 'away', team_events['home'])

    def make_row(side_key, name, m):
        row = {
            'match_id':   os.path.splitext(fname)[0],
            'date':       date_str,
            'team':       name,
            'side':       side_key,
        }
        row.update(m)
        return row

    rows = [
        make_row('home', home_name, home_m),
        make_row('away', away_name, away_m),
    ]
    return pd.DataFrame(rows)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) > 1:
        json_path = sys.argv[1]
    else:
        default = sorted(glob.glob('/home/user/Project-Beth-Mead/WSL 2024-2025/DONE/*.json'))[0]
        json_path = default
        print(f"No file specified — using: {json_path}\n")

    df = extract_metrics(json_path)

    # show metric count
    metric_cols = [c for c in df.columns if c not in ('match_id','date','team','side')]
    print(f"Metrics extracted: {len(metric_cols)}")
    print(f"Total columns: {len(df.columns)}\n")

    # print all metrics side-by-side
    pd.set_option('display.max_rows', 200)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 120)

    comp = df.set_index('team')[metric_cols].T
    print(comp.to_string())

    # optionally write CSV
    out_path = os.path.splitext(json_path)[0] + '_metrics.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}")
