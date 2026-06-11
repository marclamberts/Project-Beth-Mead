"""
extract_match_metrics.py
Extracts ~200 per-team metrics from a single Opta JSON match file.
Outputs one row per team (home / away) as a flat CSV.

Usage:
    python extract_match_metrics.py <path_to_json>
    python extract_match_metrics.py          # uses built-in demo file
"""

import json, math, sys, os, glob
from collections import defaultdict
import numpy as np
import pandas as pd

# ── pitch zone helpers ────────────────────────────────────────────────────────
def in_box(x, y):     return x > 83 and 21 < y < 79
def in_6yd(x, y):     return x > 94 and 30 < y < 70
def in_ft(x):         return x > 66
def in_mid(x):        return 33 < x <= 66
def in_own(x):        return x <= 33
def in_left(y):       return y < 33
def in_right(y):      return y > 67
def in_central(y):    return 33 <= y <= 67

def pdist(x1, y1, x2, y2):
    return math.hypot((x2 - x1) / 100 * 105, (y2 - y1) / 100 * 68)

def is_prog(x1, x2): return (x2 - x1) >= 10 and x2 > 50

# ── qualifier helpers ─────────────────────────────────────────────────────────
def qval(event, qid):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value')
    return None

def has_q(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

def sf(v, d=0.0):
    try: return float(v)
    except: return d

# ── type ID sets ──────────────────────────────────────────────────────────────
SHOT_IDS     = {13, 14, 15, 16}
ON_TGT       = {15, 16}
GOALS        = {16}
TACKLES      = {7, 51}
CLEAR        = {12, 40, 70}
AERIAL       = {44, 52}
BLOCK        = {74, 50}
RECOVERY     = {30, 49}
SAVE         = {10, 28}
DEF_ACTS     = {7, 51, 8, *CLEAR, *BLOCK}

# ─────────────────────────────────────────────────────────────────────────────

def extract_metrics(json_path: str) -> pd.DataFrame:
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)

    meta   = d.get('matchDetails', {})
    events = d['event']
    fname  = os.path.basename(json_path)
    date_str = fname.split('_')[0] if '_' in fname else ''
    tpart  = fname.replace('.json', '').split('_', 1)[1] if '_' in fname else fname
    parts  = tpart.split(' - ', 1)
    home_name = parts[0].strip() if len(parts) == 2 else 'Home'
    away_name = parts[1].strip() if len(parts) == 2 else 'Away'

    scores  = meta.get('scores', {})
    ft      = scores.get('ft', {})
    ht      = scores.get('ht', {})
    periods = {p['id']: p for p in meta.get('period', [])}
    p1_len  = periods.get(1, {}).get('lengthMin', 45)
    p2_len  = periods.get(2, {}).get('lengthMin', 45)
    p1_inj  = periods.get(1, {}).get('announcedInjuryTime', 0)
    p2_inj  = periods.get(2, {}).get('announcedInjuryTime', 0)
    mlen    = meta.get('matchLengthMin', 90)

    home_cid = away_cid = None
    for e in events:
        cid = e.get('contestantId')
        if cid and home_cid is None:
            home_cid = cid
        elif cid and cid != home_cid and away_cid is None:
            away_cid = cid
        if home_cid and away_cid:
            break

    def side(cid): return 'home' if cid == home_cid else 'away'

    te = {'home': [], 'away': []}
    for e in events:
        te[side(e.get('contestantId', ''))].append(e)

    # ── build possession sequences using Q233 (sequence ID) ──────────────────
    def build_sequences(evts):
        seqs = defaultdict(list)
        for e in evts:
            sid = qval(e, 233)
            if sid is not None:
                seqs[sid].append(e)
        return seqs

    # ── per-team metric computation ───────────────────────────────────────────
    def tm(evts, sk, opp):
        m = {}

        passes    = [e for e in evts if e.get('typeId') == 1]
        succ_pass = [e for e in passes if e.get('outcome') == 1]
        shots     = [e for e in evts if e.get('typeId') in SHOT_IDS]
        on_tgt    = [e for e in shots if e.get('typeId') in ON_TGT]
        goals     = [e for e in shots if e.get('typeId') == 16]
        tackles   = [e for e in evts if e.get('typeId') in TACKLES]
        aerials   = [e for e in evts if e.get('typeId') in AERIAL]
        dribbles  = [e for e in evts if e.get('typeId') == 3]
        opp_passes= [e for e in opp  if e.get('typeId') == 1]

        # ── PASSING (1-30) ────────────────────────────────────────────────────
        fwd=bwd=lat=into_ft=into_box=prog=0
        p_own=p_mid=p_ft=0; acc_own=acc_mid=acc_ft=0
        p_left=p_right=p_cent=0
        long_p=med_p=short_p=0
        crosses=cross_succ=0
        p_right_f=p_left_f=p_head=0
        dists=[]; through=headed_pass=free_kick_pass=throw_in_pass=counter_pass=0

        for e in passes:
            x1,y1 = sf(e.get('x'),50), sf(e.get('y'),50)
            x2 = sf(qval(e,140)); y2 = sf(qval(e,141))
            ok = e.get('outcome') == 1
            dist = pdist(x1,y1,x2,y2) if x2 and y2 else 0
            dists.append(dist)
            dx = x2-x1 if x2 else 0
            if   dx >  3: fwd+=1
            elif dx < -3: bwd+=1
            else:         lat+=1
            if x2 and in_ft(x2):          into_ft+=1
            if x2 and y2 and in_box(x2,y2): into_box+=1
            if x2 and is_prog(x1,x2):     prog+=1
            if   in_own(x1): p_own+=1; acc_own+=int(ok)
            elif in_mid(x1): p_mid+=1; acc_mid+=int(ok)
            else:            p_ft +=1; acc_ft +=int(ok)
            if   in_left(y1):    p_left+=1
            elif in_right(y1):   p_right+=1
            else:                p_cent+=1
            if   dist>32: long_p+=1
            elif dist<15: short_p+=1
            else:         med_p+=1
            bp = str(qval(e,56) or '').lower()
            if 'right' in bp: p_right_f+=1
            elif 'left' in bp: p_left_f+=1
            elif 'back' in bp: headed_pass+=1
            wide = (y1<20 or y1>80) and x2 and in_ft(x2)
            if wide:             crosses+=1
            if wide and ok:      cross_succ+=1
            if has_q(e,3):       through+=1
            if has_q(e,5):       free_kick_pass+=1
            if has_q(e,107):     throw_in_pass+=1
            if has_q(e,157):     counter_pass+=1

        m['passes_total']              = len(passes)
        m['passes_successful']         = len(succ_pass)
        m['passes_failed']             = len(passes)-len(succ_pass)
        m['pass_accuracy_pct']         = round(len(succ_pass)/max(len(passes),1)*100,1)
        m['passes_forward']            = fwd
        m['passes_backward']           = bwd
        m['passes_lateral']            = lat
        m['passes_into_final_third']   = into_ft
        m['passes_into_box']           = into_box
        m['passes_progressive']        = prog
        m['passes_own_third']          = p_own
        m['passes_mid_third']          = p_mid
        m['passes_final_third']        = p_ft
        m['passes_left_channel']       = p_left
        m['passes_right_channel']      = p_right
        m['passes_central']            = p_cent
        m['passes_long']               = long_p
        m['passes_medium']             = med_p
        m['passes_short']              = short_p
        m['crosses_total']             = crosses
        m['crosses_successful']        = cross_succ
        m['cross_accuracy_pct']        = round(cross_succ/max(crosses,1)*100,1)
        m['pass_avg_length_m']         = round(float(np.mean(dists)) if dists else 0,2)
        m['pass_max_length_m']         = round(float(np.max(dists))  if dists else 0,2)
        m['passes_right_foot']         = p_right_f
        m['passes_left_foot']          = p_left_f
        m['passes_headed']             = headed_pass
        m['pass_acc_own_third']        = round(acc_own/max(p_own,1)*100,1)
        m['pass_acc_mid_third']        = round(acc_mid/max(p_mid,1)*100,1)
        m['pass_acc_final_third']      = round(acc_ft /max(p_ft, 1)*100,1)
        m['key_passes']                = sum(1 for e in passes if has_q(e,210))
        m['through_balls']             = through
        m['through_balls_successful']  = sum(1 for e in passes if has_q(e,3) and e.get('outcome')==1)
        m['free_kick_passes']          = free_kick_pass
        m['throw_in_passes']           = throw_in_pass
        m['counterattack_passes']      = counter_pass
        m['assists']                   = sum(1 for e in evts if has_q(e,189))
        m['deep_completions']          = sum(1 for e in succ_pass if sf(qval(e,140) or 0)>83)

        # ── SHOOTING (31-60) ─────────────────────────────────────────────────
        goals_p1 = [e for e in goals if e.get('periodId')==1]
        goals_p2 = [e for e in goals if e.get('periodId')==2]
        s_box=s_6yd=s_out=0; s_rf=s_lf=s_hd=0; s_dists=[]
        sp1=sp2=0

        for e in shots:
            x,y = sf(e.get('x'),50), sf(e.get('y'),50)
            if in_6yd(x,y):    s_6yd+=1
            elif in_box(x,y):  s_box+=1
            else:              s_out+=1
            bp = str(qval(e,56) or '').lower()
            if 'right' in bp:  s_rf+=1
            elif 'left' in bp: s_lf+=1
            elif 'back' in bp: s_hd+=1
            dval = sf(qval(e,213),None)
            if dval is None:   dval = pdist(x,y,100,50)
            s_dists.append(dval)
            if e.get('periodId')==1: sp1+=1
            else:                    sp2+=1

        m['shots_total']               = len(shots)
        m['shots_on_target']           = len(on_tgt)
        m['shots_off_target']          = sum(1 for e in shots if e.get('typeId')==13)
        m['shots_post']                = sum(1 for e in shots if e.get('typeId')==14)
        m['goals']                     = len(goals)
        m['goals_first_half']          = len(goals_p1)
        m['goals_second_half']         = len(goals_p2)
        m['shot_accuracy_pct']         = round(len(on_tgt)/max(len(shots),1)*100,1)
        m['conversion_rate_pct']       = round(len(goals)/max(len(shots),1)*100,1)
        m['goals_per_shot_on_target']  = round(len(goals)/max(len(on_tgt),1)*100,1)
        m['shots_in_6yd_box']          = s_6yd
        m['shots_in_penalty_area']     = s_box
        m['shots_outside_box']         = s_out
        m['shots_right_foot']          = s_rf
        m['shots_left_foot']           = s_lf
        m['shots_headed']              = s_hd
        m['shot_avg_distance_m']       = round(float(np.mean(s_dists)) if s_dists else 0,2)
        m['shot_min_distance_m']       = round(float(np.min(s_dists))  if s_dists else 0,2)
        m['big_chances']               = sum(1 for e in evts if has_q(e,155))
        m['big_chances_scored']        = sum(1 for e in goals if has_q(e,155))
        m['big_chances_missed']        = m['big_chances'] - m['big_chances_scored']
        m['goals_from_inside_box']     = sum(1 for e in goals if in_box(sf(e.get('x')),sf(e.get('y'))))
        m['goals_outside_box']         = sum(1 for e in goals if not in_box(sf(e.get('x')),sf(e.get('y'))))
        m['goals_right_foot']          = sum(1 for e in goals if 'right' in str(qval(e,56) or '').lower())
        m['goals_left_foot']           = sum(1 for e in goals if 'left'  in str(qval(e,56) or '').lower())
        m['goals_headed']              = sum(1 for e in goals if 'back'  in str(qval(e,56) or '').lower())
        m['shots_blocked']             = sum(1 for e in evts if e.get('typeId') in BLOCK)
        m['penalties_awarded']         = sum(1 for e in evts if e.get('typeId')==56)
        m['penalty_goals']             = sum(1 for e in goals if has_q(e,9))
        m['shots_period1']             = sp1
        m['shots_period2']             = sp2
        m['goals_from_counterattack']  = sum(1 for e in goals if has_q(e,157))
        m['avg_shot_x']                = round(float(np.mean([sf(e.get('x'),50) for e in shots])) if shots else 0,2)
        m['shots_first_15min']         = sum(1 for e in shots if e.get('timeMin',0)<=15)
        m['shots_last_15min']          = sum(1 for e in shots if e.get('timeMin',0)>=75)
        m['goals_last_15min']          = sum(1 for e in goals if e.get('timeMin',0)>=75)

        # ── DEFENDING (61-85) ─────────────────────────────────────────────────
        m['tackles_total']             = len(tackles)
        m['tackles_won']               = sum(1 for e in tackles if e.get('outcome')==1)
        m['tackles_lost']              = len(tackles)-m['tackles_won']
        m['tackle_success_pct']        = round(m['tackles_won']/max(len(tackles),1)*100,1)
        m['interceptions']             = sum(1 for e in evts if e.get('typeId')==8)
        m['clearances']                = sum(1 for e in evts if e.get('typeId') in CLEAR)
        m['clearances_headed']         = sum(1 for e in evts if e.get('typeId')==40)
        m['clearances_off_line']       = sum(1 for e in evts if has_q(e,178))
        m['blocks']                    = sum(1 for e in evts if e.get('typeId') in BLOCK)
        m['aerial_duels_total']        = len(aerials)
        m['aerial_duels_won']          = sum(1 for e in aerials if e.get('outcome')==1)
        m['aerial_success_pct']        = round(m['aerial_duels_won']/max(len(aerials),1)*100,1)
        m['fouls_committed']           = sum(1 for e in evts if e.get('typeId')==4 and e.get('outcome')==0)
        m['fouls_won']                 = sum(1 for e in evts if e.get('typeId')==4 and e.get('outcome')==1)
        m['yellow_cards']              = sum(1 for e in evts if e.get('typeId')==17 and has_q(e,31))
        m['red_cards']                 = sum(1 for e in evts if e.get('typeId')==17 and has_q(e,32))
        m['gk_saves']                  = sum(1 for e in evts if e.get('typeId') in SAVE)
        m['def_actions_opp_half']      = sum(1 for e in evts if e.get('typeId') in DEF_ACTS and sf(e.get('x'),0)>50)
        m['def_actions_own_half']      = sum(1 for e in evts if e.get('typeId') in DEF_ACTS and sf(e.get('x'),0)<=50)
        m['errors_leading_shot']       = sum(1 for e in evts if e.get('typeId')==59)
        m['errors_leading_goal']       = sum(1 for e in evts if e.get('typeId')==61)
        m['offsides']                  = sum(1 for e in opp  if e.get('typeId')==5 and has_q(e,286))
        m['opp_shots_conceded']        = sum(1 for e in opp  if e.get('typeId') in SHOT_IDS)
        m['opp_goals_conceded']        = sum(1 for e in opp  if e.get('typeId')==16)
        m['opp_on_target_conceded']    = sum(1 for e in opp  if e.get('typeId') in ON_TGT)

        # ── POSSESSION / DRIBBLES (86-105) ────────────────────────────────────
        m['dribbles_total']            = len(dribbles)
        m['dribbles_successful']       = sum(1 for e in dribbles if e.get('outcome')==1)
        m['dribble_success_pct']       = round(m['dribbles_successful']/max(len(dribbles),1)*100,1)
        m['ball_recoveries']           = sum(1 for e in evts if e.get('typeId') in RECOVERY)
        m['ball_touches']              = sum(1 for e in evts if e.get('typeId')==43)
        m['total_actions']             = len(evts)
        tot_ball = len(evts)+len(opp)
        m['possession_pct']            = round(len(evts)/max(tot_ball,1)*100,1)
        m['actions_own_third']         = sum(1 for e in evts if in_own(sf(e.get('x'),50)))
        m['actions_mid_third']         = sum(1 for e in evts if in_mid(sf(e.get('x'),50)))
        m['actions_final_third']       = sum(1 for e in evts if in_ft(sf(e.get('x'),50)))
        m['actions_own_half']          = sum(1 for e in evts if sf(e.get('x'),50)<50)
        m['actions_opp_half']          = sum(1 for e in evts if sf(e.get('x'),50)>=50)
        m['opp_half_pct']              = round(m['actions_opp_half']/max(m['total_actions'],1)*100,1)
        own_def = sum(1 for e in evts if e.get('typeId') in (8,*TACKLES,4))
        opp_p   = len(opp_passes)
        m['ppda']                      = round(opp_p/max(own_def,1),2)
        m['actions_per_minute']        = round(m['total_actions']/max(mlen,1),2)
        m['actions_period1']           = sum(1 for e in evts if e.get('periodId')==1)
        m['actions_period2']           = sum(1 for e in evts if e.get('periodId')==2)

        # ── SEQUENCES (106-125) ───────────────────────────────────────────────
        seqs = build_sequences(evts)
        seq_lens = [len(v) for v in seqs.values()]
        m['sequences_total']           = len(seqs)
        m['seq_avg_length']            = round(float(np.mean(seq_lens)) if seq_lens else 0,2)
        m['seq_max_length']            = int(max(seq_lens)) if seq_lens else 0
        m['sequences_5plus_passes']    = sum(1 for v in seqs.values() if len(v)>=5)
        m['sequences_10plus_passes']   = sum(1 for v in seqs.values() if len(v)>=10)
        # sequences ending in shot
        seq_shot=seq_goal=seq_box_entry=0
        seq_start_xs=[]
        for sid,sv in seqs.items():
            tids = {e.get('typeId') for e in sv}
            xs   = [sf(e.get('x'),50) for e in sv if e.get('x')]
            if xs: seq_start_xs.append(xs[0])
            if tids & SHOT_IDS:   seq_shot+=1
            if tids & GOALS:      seq_goal+=1
            if any(in_box(sf(e.get('x'),50),sf(e.get('y'),50)) for e in sv): seq_box_entry+=1
        m['sequences_ending_shot']     = seq_shot
        m['sequences_ending_goal']     = seq_goal
        m['sequences_box_entry']       = seq_box_entry
        m['seq_avg_start_x']           = round(float(np.mean(seq_start_xs)) if seq_start_xs else 0,2)
        m['passes_per_sequence']       = round(len(passes)/max(len(seqs),1),2)

        # ── SET PIECES (126-140) ──────────────────────────────────────────────
        corners    = [e for e in evts if e.get('typeId')==6]
        free_kicks = [e for e in evts if e.get('typeId')==1 and has_q(e,5)]
        throw_ins  = [e for e in evts if e.get('typeId')==1 and has_q(e,107)]
        goal_kicks = [e for e in evts if e.get('typeId')==41]

        m['corners_taken']             = len(corners)
        m['corners_conceded']          = sum(1 for e in opp if e.get('typeId')==6)
        m['free_kicks_taken']          = len(free_kicks)
        m['throw_ins_taken']           = len(throw_ins)
        m['throw_in_success_pct']      = round(sum(1 for e in throw_ins if e.get('outcome')==1)/max(len(throw_ins),1)*100,1)
        m['goal_kicks']                = len(goal_kicks)
        m['direct_free_kicks']         = sum(1 for e in evts if e.get('typeId')==45)
        m['set_piece_goals']           = sum(1 for e in goals if has_q(e,9) or has_q(e,286) or has_q(e,5))
        m['goals_from_corners']        = sum(1 for e in goals if has_q(e,286))
        m['goals_from_free_kicks']     = sum(1 for e in goals if has_q(e,5))
        m['shots_from_set_piece']      = sum(1 for e in shots if has_q(e,5) or has_q(e,286))
        m['gk_throws']                 = sum(1 for e in evts if e.get('typeId')==27)
        m['gk_distribution_total']     = sum(1 for e in evts if e.get('typeId') in (27,34,41))
        m['penalties_scored']          = m['penalty_goals']
        m['penalties_conceded']        = sum(1 for e in opp if e.get('typeId')==56)

        # ── TEMPO / INTENSITY (141-155) ───────────────────────────────────────
        first_shot = min((e.get('timeMin',999) for e in shots), default=None)
        first_goal = min((e.get('timeMin',999) for e in goals), default=None)
        m['first_shot_minute']         = first_shot if first_shot is not None else -1
        m['first_goal_minute']         = first_goal if first_goal is not None else -1
        m['turnovers_opp_half']        = sum(1 for e in evts if e.get('typeId')==1 and e.get('outcome')!=1 and sf(e.get('x'),0)>50)
        m['turnovers_own_half']        = sum(1 for e in evts if e.get('typeId')==1 and e.get('outcome')!=1 and sf(e.get('x'),0)<=50)
        m['counterattack_actions']     = sum(1 for e in evts if has_q(e,157))
        m['passes_under_pressure']     = sum(1 for e in passes if has_q(e,2))
        m['pass_acc_under_pressure']   = round(sum(1 for e in passes if has_q(e,2) and e.get('outcome')==1)/max(sum(1 for e in passes if has_q(e,2)),1)*100,1)
        m['high_press_recoveries']     = sum(1 for e in evts if e.get('typeId') in RECOVERY and sf(e.get('x'),0)>50)
        m['avg_pass_start_x']          = round(float(np.mean([sf(e.get('x'),50) for e in passes])) if passes else 0,2)

        # ── PLAYER STATS (156-175) ────────────────────────────────────────────
        p_acts=defaultdict(int); p_passes=defaultdict(int); p_shots=defaultdict(int)
        p_goals=defaultdict(int); p_tackles=defaultdict(int); p_dribbles=defaultdict(int)
        p_aerials=defaultdict(int); p_interceptions=defaultdict(int)
        for e in evts:
            pn = e.get('playerName') or ''
            if not pn: continue
            p_acts[pn]+=1
            tid = e.get('typeId')
            if tid==1:            p_passes[pn]+=1
            if tid in SHOT_IDS:   p_shots[pn]+=1
            if tid==16:           p_goals[pn]+=1
            if tid in TACKLES:    p_tackles[pn]+=1
            if tid==3:            p_dribbles[pn]+=1
            if tid in AERIAL:     p_aerials[pn]+=1
            if tid==8:            p_interceptions[pn]+=1

        def top(d, default_val=0):
            if not d: return ('', default_val)
            k = max(d, key=d.get)
            return (k, d[k])

        top_act, top_act_n   = top(p_acts)
        top_pas, top_pas_n   = top(p_passes)
        top_sh,  top_sh_n    = top(p_shots)
        top_g,   top_g_n     = top(p_goals)
        top_tk,  top_tk_n    = top(p_tackles)
        top_dr,  top_dr_n    = top(p_dribbles)
        top_ae,  top_ae_n    = top(p_aerials)
        top_int, top_int_n   = top(p_interceptions)

        m['unique_players']            = len(set(e.get('playerName') for e in evts if e.get('playerName')))
        m['substitutions_made']        = sum(1 for e in evts if e.get('typeId')==18)
        m['top_player_by_actions']     = top_act
        m['top_player_action_count']   = top_act_n
        m['top_passer']                = top_pas
        m['top_passer_count']          = top_pas_n
        m['top_shooter']               = top_sh
        m['top_shooter_shots']         = top_sh_n
        m['top_scorer']                = top_g
        m['top_scorer_goals']          = top_g_n
        m['top_tackler']               = top_tk
        m['top_tackler_count']         = top_tk_n
        m['top_dribbler']              = top_dr
        m['top_dribbler_count']        = top_dr_n
        m['top_aerial_player']         = top_ae
        m['top_aerial_count']          = top_ae_n
        m['top_interceptor']           = top_int
        m['top_interceptor_count']     = top_int_n

        # ── RESULT / MATCH CONTEXT (176-190) ─────────────────────────────────
        opp_side = 'away' if sk=='home' else 'home'
        m['result']                    = 'win' if meta.get('winner')==sk else ('draw' if meta.get('winner')=='draw' else 'loss')
        m['goals_scored']              = ft.get(sk, 0)
        m['goals_conceded']            = ft.get(opp_side, 0)
        m['ht_goals_scored']           = ht.get(sk, 0)
        m['ht_goals_conceded']         = ht.get(opp_side, 0)
        m['goal_difference']           = m['goals_scored']-m['goals_conceded']
        m['clean_sheet']               = int(m['goals_conceded']==0)
        m['came_from_behind']          = int(m['ht_goals_scored']<m['ht_goals_conceded'] and m['result']=='win')
        m['threw_away_lead']           = int(m['ht_goals_scored']>m['ht_goals_conceded'] and m['result']=='loss')
        m['shots_ratio_pct']           = round(len(shots)/max(len(shots)+sum(1 for e in opp if e.get('typeId') in SHOT_IDS),1)*100,1)
        m['on_target_ratio_pct']       = round(len(on_tgt)/max(len(on_tgt)+m['opp_on_target_conceded'],1)*100,1)

        # ── ADVANCED RATIOS (191-205) ─────────────────────────────────────────
        m['shots_per_pass']            = round(len(shots)/max(len(passes),1),4)
        m['progressive_pass_pct']      = round(prog/max(len(passes),1)*100,1)
        m['passes_into_box_pct']       = round(into_box/max(len(passes),1)*100,1)
        m['pass_into_ft_pct']          = round(into_ft/max(len(passes),1)*100,1)
        m['def_vs_opp_pass_ratio']     = round(own_def/max(opp_p,1),4)
        m['recovery_rate_pct']         = round(m['ball_recoveries']/max(m['total_actions'],1)*100,2)
        m['attack_third_pass_pct']     = round(p_ft/max(len(passes),1)*100,1)
        m['pressure_index']            = round(m['def_actions_opp_half']/max(m['total_actions'],1)*100,2)
        m['shot_quality_index']        = round(m['shot_accuracy_pct']*m['conversion_rate_pct']/100,2)
        m['box_entry_rate_pct']        = round(m['sequences_box_entry']/max(m['sequences_total'],1)*100,1)
        m['shot_conversion_seq_pct']   = round(m['sequences_ending_shot']/max(m['sequences_total'],1)*100,1)
        m['cross_to_shot_pct']         = round(sum(1 for e in shots if any(e2.get('typeId')==1 and (sf(e2.get('y'),50)<20 or sf(e2.get('y'),50)>80) for e2 in evts))/max(len(shots),1)*100,1)
        m['long_pass_accuracy_pct']    = round(sum(1 for e in passes if pdist(sf(e.get('x'),50),sf(e.get('y'),50),sf(qval(e,140) or 50,50),sf(qval(e,141) or 50,50))>32 and e.get('outcome')==1)/max(long_p,1)*100,1)
        m['dribble_to_shot_pct']       = round(m['dribbles_successful']/max(len(shots),1)*100,1)
        m['aerial_dominance_index']    = round(m['aerial_duels_won']/max(sum(1 for e in opp if e.get('typeId') in AERIAL),1)*100,1)

        # ── MATCH META (206-210) ──────────────────────────────────────────────
        m['match_length_min']          = mlen
        m['period1_length_min']        = p1_len
        m['period2_length_min']        = p2_len
        m['period1_injury_time_s']     = p1_inj
        m['period2_injury_time_s']     = p2_inj

        return m

    hm = tm(te['home'], 'home', te['away'])
    am = tm(te['away'], 'away', te['home'])

    rows = []
    for sk, name, metrics in [('home', home_name, hm), ('away', away_name, am)]:
        row = {'match_id': os.path.splitext(fname)[0], 'date': date_str,
               'team': name, 'side': sk}
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


# ── run on all files → master CSV ─────────────────────────────────────────────

def build_all_matches(output_csv='all_match_metrics.csv'):
    files = sorted(glob.glob('/home/user/Project-Beth-Mead/WSL*/**/*.json', recursive=True)
                 + glob.glob('/home/user/Project-Beth-Mead/WSL*/*.json'))
    print(f"Processing {len(files)} files...")
    frames = []
    for i, fp in enumerate(files):
        if i % 100 == 0:
            print(f"  {i}/{len(files)}")
        try:
            frames.append(extract_metrics(fp))
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} ({len(df):,} rows × {len(df.columns)} columns)")
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        build_all_matches()
    else:
        if len(sys.argv) > 1:
            json_path = sys.argv[1]
        else:
            json_path = sorted(glob.glob('/home/user/Project-Beth-Mead/WSL 2024-2025/DONE/*.json'))[0]
            print(f"Using: {json_path}\n")

        df = extract_metrics(json_path)
        metric_cols = [c for c in df.columns if c not in ('match_id','date','team','side')]
        print(f"Metrics: {len(metric_cols)}\n")
        pd.set_option('display.max_rows', 300)
        print(df.set_index('team')[metric_cols].T.to_string())

        out = os.path.splitext(json_path)[0] + '_metrics.csv'
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")
