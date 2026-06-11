"""
extract_player_metrics.py
Extracts 250 per-player metrics from a single Opta JSON match file.
xG data is joined from the matching xgCSV file where available.
Output: one row per player, 250 metric columns.

Usage:
    python extract_player_metrics.py <path_to_json>   # single match
    python extract_player_metrics.py --all             # all matches → all_player_metrics.csv
"""

import json, math, sys, os, glob
from collections import defaultdict
import numpy as np
import pandas as pd

# ── pitch zone helpers ────────────────────────────────────────────────────────
def in_box(x, y):    return x > 83 and 21 < y < 79
def in_6yd(x, y):    return x > 94 and 30 < y < 70
def in_ft(x):        return x > 66
def in_mid(x):       return 33 < x <= 66
def in_own(x):       return x <= 33
def in_left(y):      return y < 33
def in_right(y):     return y > 67
def in_central(y):   return 33 <= y <= 67

def pdist(x1, y1, x2, y2):
    return math.hypot((x2 - x1) / 100 * 105, (y2 - y1) / 100 * 68)

def is_prog(x1, x2): return (x2 - x1) >= 10 and x2 > 50

def sf(v, d=0.0):
    try:    return float(v)
    except: return d

def qval(event, qid):
    for q in event.get('qualifier', []):
        if q['qualifierId'] == qid:
            return q.get('value')
    return None

def has_q(event, qid):
    return any(q['qualifierId'] == qid for q in event.get('qualifier', []))

# ── type ID constants ─────────────────────────────────────────────────────────
PASS_ID     = 1
DRIBBLE_ID  = 3
FOUL_ID     = 4
CORNER_ID   = 6
TACKLE_IDS  = {7, 51}
INTERC_ID   = 8
SAVE_IDS    = {10, 28}
CLAIM_ID    = 11
CLEAR_IDS   = {12, 40, 70}
SHOT_IDS    = {13, 14, 15, 16}
ON_TGT      = {15, 16}
GOAL_ID     = 16
CARD_ID     = 17
SUB_OFF     = 18
SUB_ON      = 19
GK_THROW_ID = 27
GK_KICK_ID  = 41
TOUCH_ID    = 43
AERIAL_IDS  = {44, 52}
DFREE_ID    = 45
RECOVERY_ID = {30, 49}
BLK_IDS     = {50, 74}
PEN_ID      = 56
ERR_SHOT_ID = 59
ERR_GOAL_ID = 61


def _find_xg_csv(json_path: str) -> pd.DataFrame:
    """Locate the matching xgCSV file and return it, or empty DataFrame."""
    base_dir = os.path.dirname(os.path.dirname(json_path))
    fname    = os.path.splitext(os.path.basename(json_path))[0]
    xg_path  = os.path.join(base_dir, "xgCSV", fname + ".csv")
    if not os.path.exists(xg_path):
        # try recursive search
        hits = glob.glob(os.path.join(os.path.dirname(json_path), "**", fname + ".csv"), recursive=True)
        if not hits:
            return pd.DataFrame()
        xg_path = hits[0]
    try:
        return pd.read_csv(xg_path)
    except Exception:
        return pd.DataFrame()


def extract_player_metrics(json_path: str) -> pd.DataFrame:
    with open(json_path, encoding='utf-8') as f:
        d = json.load(f)

    events   = d['event']
    meta     = d.get('matchDetails', {})
    fname    = os.path.basename(json_path)
    date_str = fname.split('_')[0] if '_' in fname else ''
    tpart    = fname.replace('.json', '').split('_', 1)[1] if '_' in fname else fname
    parts    = tpart.split(' - ', 1)
    home_name = parts[0].strip() if len(parts) == 2 else 'Home'
    away_name = parts[1].strip() if len(parts) == 2 else 'Away'

    home_cid = away_cid = None
    for e in events:
        cid = e.get('contestantId')
        if cid and home_cid is None:            home_cid = cid
        elif cid and cid != home_cid and away_cid is None: away_cid = cid
        if home_cid and away_cid: break

    def team_name(cid):
        return home_name if cid == home_cid else away_name

    # ── group events by player ────────────────────────────────────────────────
    player_events = defaultdict(list)
    for e in events:
        pn = e.get('playerName')
        if pn:
            player_events[pn].append(e)

    # ── load xG CSV ───────────────────────────────────────────────────────────
    xg_df = _find_xg_csv(json_path)
    xg_by_player = {}
    if not xg_df.empty and 'PlayerId' in xg_df.columns:
        for pname, grp in xg_df.groupby('PlayerId'):
            xg_by_player[pname] = grp

    # ── per-player metric computation ─────────────────────────────────────────
    def player_metrics(pname, evts):
        m = {}
        cid   = evts[0].get('contestantId', '') if evts else ''
        tm    = team_name(cid)
        side  = 'home' if cid == home_cid else 'away'

        # event type slices
        passes    = [e for e in evts if e.get('typeId') == PASS_ID]
        succ_p    = [e for e in passes if e.get('outcome') == 1]
        shots     = [e for e in evts if e.get('typeId') in SHOT_IDS]
        on_tgt    = [e for e in shots  if e.get('typeId') in ON_TGT]
        goals     = [e for e in shots  if e.get('typeId') == GOAL_ID]
        dribbles  = [e for e in evts if e.get('typeId') == DRIBBLE_ID]
        tackles   = [e for e in evts if e.get('typeId') in TACKLE_IDS]
        aerials   = [e for e in evts if e.get('typeId') in AERIAL_IDS]
        clearances= [e for e in evts if e.get('typeId') in CLEAR_IDS]
        blocks    = [e for e in evts if e.get('typeId') in BLK_IDS]
        recoveries= [e for e in evts if e.get('typeId') in RECOVERY_ID]
        touches   = [e for e in evts if e.get('typeId') == TOUCH_ID]

        # ── PASSING (1-60) ────────────────────────────────────────────────────
        fwd=bwd=lat=0; into_ft=into_box=prog=0
        p_own=p_mid=p_ft=0; acc_own=acc_mid=acc_ft=0
        p_left=p_right=p_cent=0; acc_left=acc_right=acc_cent=0
        long_p=med_p=short_p=0; acc_long=acc_short=0
        p_rf=p_lf=p_hd=0; acc_rf=acc_lf=0
        crosses=cross_succ=cross_left=cross_right=0
        through=through_succ=0; kp=dp=0
        fp=ti=ti_succ=cp=0; pup=pup_succ=0
        dists=[]; start_xs=[]

        for e in passes:
            x1,y1 = sf(e.get('x'),50), sf(e.get('y'),50)
            x2 = sf(qval(e,140)); y2 = sf(qval(e,141))
            ok = e.get('outcome') == 1
            dist = pdist(x1,y1,x2,y2) if x2 and y2 else 0
            dists.append(dist); start_xs.append(x1)
            dx = x2-x1 if x2 else 0
            if   dx >  3: fwd+=1
            elif dx < -3: bwd+=1
            else:         lat+=1
            if x2 and in_ft(x2):             into_ft+=1
            if x2 and y2 and in_box(x2,y2):  into_box+=1
            if x2 and is_prog(x1,x2):        prog+=1
            if   in_own(x1): p_own+=1; acc_own+=int(ok)
            elif in_mid(x1): p_mid+=1; acc_mid+=int(ok)
            else:            p_ft +=1; acc_ft +=int(ok)
            if   in_left(y1):   p_left+=1;  acc_left+=int(ok)
            elif in_right(y1):  p_right+=1; acc_right+=int(ok)
            else:               p_cent+=1;  acc_cent+=int(ok)
            if   dist>32: long_p+=1;  acc_long+=int(ok)
            elif dist<15: short_p+=1; acc_short+=int(ok)
            else:         med_p+=1
            bp = str(qval(e,56) or '').lower()
            if 'right' in bp:  p_rf+=1; acc_rf+=int(ok)
            elif 'left' in bp: p_lf+=1; acc_lf+=int(ok)
            elif 'back' in bp: p_hd+=1
            wide = (y1<20 or y1>80)
            if wide and x2 and in_ft(x2):
                crosses+=1
                if ok: cross_succ+=1
                if in_left(y1): cross_left+=1
                else:            cross_right+=1
            if has_q(e,3):   through+=1;  through_succ+=int(ok)
            if has_q(e,210): kp+=1
            if x2 and sf(x2)>83: dp+=1
            if has_q(e,5):   fp+=1
            if has_q(e,107): ti+=1; ti_succ+=int(ok)
            if has_q(e,157): cp+=1
            if has_q(e,2):   pup+=1; pup_succ+=int(ok)

        n_p = len(passes); n_sp = len(succ_p)
        m['passes_total']                = n_p
        m['passes_successful']           = n_sp
        m['passes_failed']               = n_p - n_sp
        m['pass_accuracy_pct']           = round(n_sp/max(n_p,1)*100,1)
        m['passes_forward']              = fwd
        m['passes_backward']             = bwd
        m['passes_lateral']              = lat
        m['pass_forward_accuracy_pct']   = round(sum(1 for e in passes if sf(qval(e,140),0)-sf(e.get('x'),50)>3 and e.get('outcome')==1)/max(fwd,1)*100,1)
        m['passes_own_third']            = p_own
        m['passes_mid_third']            = p_mid
        m['passes_final_third']          = p_ft
        m['pass_acc_own_third']          = round(acc_own/max(p_own,1)*100,1)
        m['pass_acc_mid_third']          = round(acc_mid/max(p_mid,1)*100,1)
        m['pass_acc_final_third']        = round(acc_ft /max(p_ft, 1)*100,1)
        m['passes_left_channel']         = p_left
        m['passes_right_channel']        = p_right
        m['passes_central']              = p_cent
        m['pass_acc_left_channel']       = round(acc_left /max(p_left, 1)*100,1)
        m['pass_acc_right_channel']      = round(acc_right/max(p_right,1)*100,1)
        m['pass_acc_central']            = round(acc_cent /max(p_cent, 1)*100,1)
        m['passes_long']                 = long_p
        m['passes_medium']               = med_p
        m['passes_short']                = short_p
        m['long_pass_accuracy_pct']      = round(acc_long /max(long_p, 1)*100,1)
        m['short_pass_accuracy_pct']     = round(acc_short/max(short_p,1)*100,1)
        m['pass_avg_length_m']           = round(float(np.mean(dists)) if dists else 0,2)
        m['pass_max_length_m']           = round(float(np.max(dists))  if dists else 0,2)
        m['pass_min_length_m']           = round(float(np.min(dists))  if dists else 0,2)
        m['pass_total_distance_m']       = round(sum(dists),1)
        m['passes_right_foot']           = p_rf
        m['passes_left_foot']            = p_lf
        m['passes_headed']               = p_hd
        m['pass_right_foot_accuracy_pct']= round(acc_rf/max(p_rf,1)*100,1)
        m['pass_left_foot_accuracy_pct'] = round(acc_lf/max(p_lf,1)*100,1)
        m['passes_into_final_third']     = into_ft
        m['passes_into_box']             = into_box
        m['passes_progressive']          = prog
        m['progressive_pass_pct']        = round(prog  /max(n_p,1)*100,1)
        m['passes_into_final_third_pct'] = round(into_ft/max(n_p,1)*100,1)
        m['passes_into_box_pct']         = round(into_box/max(n_p,1)*100,1)
        m['key_passes']                  = kp
        m['key_pass_rate_pct']           = round(kp/max(n_p,1)*100,2)
        m['through_balls']               = through
        m['through_balls_successful']    = through_succ
        m['through_ball_accuracy_pct']   = round(through_succ/max(through,1)*100,1)
        m['crosses_total']               = crosses
        m['crosses_successful']          = cross_succ
        m['cross_accuracy_pct']          = round(cross_succ/max(crosses,1)*100,1)
        m['crosses_left_channel']        = cross_left
        m['crosses_right_channel']       = cross_right
        m['deep_completions']            = dp
        m['passes_under_pressure']       = pup
        m['pass_acc_under_pressure_pct'] = round(pup_succ/max(pup,1)*100,1)
        m['free_kick_passes']            = fp
        m['throw_in_passes']             = ti
        m['throw_in_success_pct']        = round(ti_succ/max(ti,1)*100,1)
        m['counterattack_passes']        = cp
        m['passes_period1']              = sum(1 for e in passes if e.get('periodId')==1)
        m['passes_period2']              = sum(1 for e in passes if e.get('periodId')==2)
        m['avg_pass_start_x']            = round(float(np.mean(start_xs)) if start_xs else 0,2)

        # ── SHOOTING (61-110) ─────────────────────────────────────────────────
        s_rf=s_lf=s_hd=0; s_6yd=s_box=s_out=0; s_dists=[]; s_xs=[]; s_ys=[]
        g_box=g_out=g_rf=g_lf=g_hd=0; s_op=s_sp=s_ca=0; pens=pen_g=0
        big_c=big_g=0

        for e in shots:
            x,y = sf(e.get('x'),50), sf(e.get('y'),50)
            bp  = str(qval(e,56) or '').lower()
            ok  = e.get('typeId') == GOAL_ID
            s_xs.append(x); s_ys.append(y)
            dist = sf(qval(e,213), pdist(x,y,100,50))
            s_dists.append(dist)
            if   in_6yd(x,y):  s_6yd+=1
            elif in_box(x,y):  s_box+=1
            else:              s_out+=1
            if 'right' in bp:  s_rf+=1
            elif 'left' in bp: s_lf+=1
            elif 'back' in bp: s_hd+=1
            if ok:
                if in_box(x,y): g_box+=1
                else:            g_out+=1
                if 'right' in bp: g_rf+=1
                elif 'left' in bp: g_lf+=1
                elif 'back' in bp: g_hd+=1
            if has_q(e,155): big_c+=1
            if has_q(e,155) and ok: big_g+=1
            if has_q(e,9):   pens+=1; pen_g+=int(ok)
            if has_q(e,157): s_ca+=1
            if has_q(e,5) or has_q(e,286): s_sp+=1
            else: s_op+=1

        n_sh=len(shots); n_g=len(goals); n_ot=len(on_tgt)
        m['shots_total']                 = n_sh
        m['shots_on_target']             = n_ot
        m['shots_off_target']            = sum(1 for e in shots if e.get('typeId')==13)
        m['shots_post']                  = sum(1 for e in shots if e.get('typeId')==14)
        m['shots_blocked']               = sum(1 for e in evts  if e.get('typeId') in BLK_IDS and sf(e.get('x'),50)>50)
        m['goals']                       = n_g
        m['shot_accuracy_pct']           = round(n_ot/max(n_sh,1)*100,1)
        m['conversion_rate_pct']         = round(n_g /max(n_sh,1)*100,1)
        m['goals_per_shot_on_target_pct']= round(n_g /max(n_ot,1)*100,1)
        m['shots_right_foot']            = s_rf
        m['shots_left_foot']             = s_lf
        m['shots_headed']                = s_hd
        m['shots_in_6yd_box']            = s_6yd
        m['shots_in_penalty_area']       = s_box
        m['shots_outside_box']           = s_out
        m['goals_inside_box']            = g_box
        m['goals_outside_box']           = g_out
        m['goals_right_foot']            = g_rf
        m['goals_left_foot']             = g_lf
        m['goals_headed']                = g_hd
        m['shot_avg_distance_m']         = round(float(np.mean(s_dists)) if s_dists else 0,2)
        m['shot_min_distance_m']         = round(float(np.min(s_dists))  if s_dists else 0,2)
        m['shot_max_distance_m']         = round(float(np.max(s_dists))  if s_dists else 0,2)
        m['shot_avg_x']                  = round(float(np.mean(s_xs)) if s_xs else 0,2)
        m['shot_avg_y']                  = round(float(np.mean(s_ys)) if s_ys else 0,2)
        m['shots_period1']               = sum(1 for e in shots if e.get('periodId')==1)
        m['shots_period2']               = sum(1 for e in shots if e.get('periodId')==2)
        m['shots_first_15min']           = sum(1 for e in shots if e.get('timeMin',0)<=15)
        m['shots_last_15min']            = sum(1 for e in shots if e.get('timeMin',0)>=75)
        m['big_chances']                 = big_c
        m['big_chances_scored']          = big_g
        m['big_chances_missed']          = big_c - big_g
        m['big_chance_conversion_pct']   = round(big_g/max(big_c,1)*100,1)
        m['shots_open_play']             = s_op
        m['shots_set_piece']             = s_sp
        m['shots_counterattack']         = s_ca
        m['penalties_taken']             = pens
        m['penalties_scored']            = pen_g

        # xG from CSV
        xg_p = xg_by_player.get(pname, pd.DataFrame())
        if not xg_p.empty:
            m['xG_total']                = round(float(xg_p['xG'].sum()), 4)
            m['xG_per_shot']             = round(float(xg_p['xG'].mean()), 4)
            m['goals_minus_xG']          = round(n_g - m['xG_total'], 4)
            m['xG_on_target']            = round(float(xg_p[xg_p.get('isGoal', pd.Series()) != False]['xG'].sum()) if 'isGoal' in xg_p.columns else m['xG_total'], 4)
            ot_xg = xg_p[xg_p['isGoal'] == True]['xG'] if 'isGoal' in xg_p.columns else pd.Series()
            m['xG_on_target']            = round(float(ot_xg.sum()) if len(ot_xg) else m['xG_total'], 4)
            sp_mask = xg_p['Type_of_play'].isin(['DirectFreekick','Penalty','Corner']) if 'Type_of_play' in xg_p.columns else pd.Series([False]*len(xg_p))
            m['xG_open_play']            = round(float(xg_p[~sp_mask]['xG'].sum()), 4)
            m['xG_set_piece']            = round(float(xg_p[sp_mask]['xG'].sum()), 4)
            m['shot_distance_avg_csv']   = round(float(xg_p['distance'].mean()), 2) if 'distance' in xg_p.columns else 0.0
            m['avg_shot_angle']          = round(float(xg_p['angle'].mean()), 2)  if 'angle'    in xg_p.columns else 0.0
            m['shots_assisted']          = int((xg_p['isAssistedShot'].astype(str).str.strip().str.lower()=='yes').sum()) if 'isAssistedShot' in xg_p.columns else 0
            m['shots_regular_play']      = int((xg_p['Type_of_play']=='RegularPlay').sum()) if 'Type_of_play' in xg_p.columns else 0
        else:
            for col in ['xG_total','xG_per_shot','goals_minus_xG','xG_on_target',
                        'xG_open_play','xG_set_piece','shot_distance_avg_csv',
                        'avg_shot_angle','shots_assisted','shots_regular_play']:
                m[col] = 0.0

        # ── DRIBBLING / CARRYING (111-135) ────────────────────────────────────
        dr_rf=dr_lf=0; dr_own=dr_mid=dr_ft=0; dr_left=dr_right=dr_cent=0

        for e in dribbles:
            x,y = sf(e.get('x'),50), sf(e.get('y'),50)
            if   in_own(x):    dr_own+=1
            elif in_mid(x):    dr_mid+=1
            else:              dr_ft+=1
            if   in_left(y):   dr_left+=1
            elif in_right(y):  dr_right+=1
            else:              dr_cent+=1

        n_dr = len(dribbles); dr_succ = sum(1 for e in dribbles if e.get('outcome')==1)
        m['dribbles_total']              = n_dr
        m['dribbles_successful']         = dr_succ
        m['dribble_success_pct']         = round(dr_succ/max(n_dr,1)*100,1)
        m['dribbles_own_third']          = dr_own
        m['dribbles_mid_third']          = dr_mid
        m['dribbles_final_third']        = dr_ft
        m['dribbles_left_channel']       = dr_left
        m['dribbles_right_channel']      = dr_right
        m['dribbles_central']            = dr_cent
        m['dribbles_period1']            = sum(1 for e in dribbles if e.get('periodId')==1)
        m['dribbles_period2']            = sum(1 for e in dribbles if e.get('periodId')==2)
        m['dribbles_in_box']             = sum(1 for e in dribbles if in_box(sf(e.get('x'),50),sf(e.get('y'),50)))
        m['dribble_acc_final_third']     = round(sum(1 for e in dribbles if in_ft(sf(e.get('x'),50)) and e.get('outcome')==1)/max(dr_ft,1)*100,1)
        m['dribbles_counterattack']      = sum(1 for e in dribbles if has_q(e,157))
        m['dribbles_per_shot']           = round(n_dr/max(n_sh,1),3)

        carries_fwd = sum(1 for e in touches if sf(e.get('x'),50)>50)
        m['ball_touches']                = len(touches)
        m['touches_own_third']           = sum(1 for e in touches if in_own(sf(e.get('x'),50)))
        m['touches_mid_third']           = sum(1 for e in touches if in_mid(sf(e.get('x'),50)))
        m['touches_final_third']         = sum(1 for e in touches if in_ft(sf(e.get('x'),50)))
        m['touches_in_box']              = sum(1 for e in touches if in_box(sf(e.get('x'),50),sf(e.get('y'),50)))
        m['carries_forward']             = carries_fwd
        m['carries_opp_half']            = sum(1 for e in touches if sf(e.get('x'),50)>=50)
        m['progressive_actions']         = prog + dr_succ + carries_fwd

        # ── DEFENDING (136-180) ───────────────────────────────────────────────
        n_tk = len(tackles); tk_w = sum(1 for e in tackles if e.get('outcome')==1)
        m['tackles_total']               = n_tk
        m['tackles_won']                 = tk_w
        m['tackles_lost']                = n_tk - tk_w
        m['tackle_success_pct']          = round(tk_w/max(n_tk,1)*100,1)
        m['tackles_own_third']           = sum(1 for e in tackles if in_own(sf(e.get('x'),50)))
        m['tackles_mid_third']           = sum(1 for e in tackles if in_mid(sf(e.get('x'),50)))
        m['tackles_final_third']         = sum(1 for e in tackles if in_ft(sf(e.get('x'),50)))
        m['tackles_period1']             = sum(1 for e in tackles if e.get('periodId')==1)
        m['tackles_period2']             = sum(1 for e in tackles if e.get('periodId')==2)
        m['interceptions_total']         = sum(1 for e in evts if e.get('typeId')==INTERC_ID)
        m['interceptions_own_third']     = sum(1 for e in evts if e.get('typeId')==INTERC_ID and in_own(sf(e.get('x'),50)))
        m['interceptions_mid_third']     = sum(1 for e in evts if e.get('typeId')==INTERC_ID and in_mid(sf(e.get('x'),50)))
        m['interceptions_final_third']   = sum(1 for e in evts if e.get('typeId')==INTERC_ID and in_ft(sf(e.get('x'),50)))
        m['clearances_total']            = len(clearances)
        m['clearances_headed']           = sum(1 for e in evts if e.get('typeId')==40)
        m['clearances_off_line']         = sum(1 for e in evts if has_q(e,178))
        m['clearances_own_third']        = sum(1 for e in clearances if in_own(sf(e.get('x'),50)))
        m['clearances_mid_third']        = sum(1 for e in clearances if in_mid(sf(e.get('x'),50)))
        m['blocks_total']                = len(blocks)
        m['blocks_shots']                = sum(1 for e in evts if e.get('typeId')==74)
        m['blocks_passes']               = sum(1 for e in evts if e.get('typeId')==50)
        n_ae = len(aerials); ae_w = sum(1 for e in aerials if e.get('outcome')==1)
        m['aerial_duels_total']          = n_ae
        m['aerial_duels_won']            = ae_w
        m['aerial_duels_lost']           = n_ae - ae_w
        m['aerial_success_pct']          = round(ae_w/max(n_ae,1)*100,1)
        m['aerial_offensive']            = sum(1 for e in aerials if sf(e.get('x'),50)>=50)
        m['aerial_defensive']            = sum(1 for e in aerials if sf(e.get('x'),50)<50)
        m['aerial_won_own_half']         = sum(1 for e in aerials if e.get('outcome')==1 and sf(e.get('x'),50)<50)
        m['aerial_won_opp_half']         = sum(1 for e in aerials if e.get('outcome')==1 and sf(e.get('x'),50)>=50)
        m['fouls_committed']             = sum(1 for e in evts if e.get('typeId')==FOUL_ID and e.get('outcome')==0)
        m['fouls_committed_own_third']   = sum(1 for e in evts if e.get('typeId')==FOUL_ID and e.get('outcome')==0 and in_own(sf(e.get('x'),50)))
        m['fouls_committed_mid_third']   = sum(1 for e in evts if e.get('typeId')==FOUL_ID and e.get('outcome')==0 and in_mid(sf(e.get('x'),50)))
        m['fouls_committed_final_third'] = sum(1 for e in evts if e.get('typeId')==FOUL_ID and e.get('outcome')==0 and in_ft(sf(e.get('x'),50)))
        m['fouls_won']                   = sum(1 for e in evts if e.get('typeId')==FOUL_ID and e.get('outcome')==1)
        m['yellow_cards']                = sum(1 for e in evts if e.get('typeId')==CARD_ID and has_q(e,31))
        m['red_cards']                   = sum(1 for e in evts if e.get('typeId')==CARD_ID and has_q(e,32))
        m['errors_leading_shot']         = sum(1 for e in evts if e.get('typeId')==ERR_SHOT_ID)
        m['errors_leading_goal']         = sum(1 for e in evts if e.get('typeId')==ERR_GOAL_ID)
        def_total = n_tk + m['interceptions_total'] + len(clearances) + len(blocks)
        m['defensive_actions_total']     = def_total
        m['def_actions_own_third']       = sum(1 for e in evts if e.get('typeId') in (TACKLE_IDS|{INTERC_ID}|CLEAR_IDS|BLK_IDS) and in_own(sf(e.get('x'),50)))
        m['def_actions_mid_third']       = sum(1 for e in evts if e.get('typeId') in (TACKLE_IDS|{INTERC_ID}|CLEAR_IDS|BLK_IDS) and in_mid(sf(e.get('x'),50)))
        m['def_actions_final_third']     = sum(1 for e in evts if e.get('typeId') in (TACKLE_IDS|{INTERC_ID}|CLEAR_IDS|BLK_IDS) and in_ft(sf(e.get('x'),50)))
        m['pressures_opp_half']          = sum(1 for e in evts if e.get('typeId') in (TACKLE_IDS|{INTERC_ID,FOUL_ID}) and sf(e.get('x'),50)>=50)
        m['duels_total']                 = n_tk + n_ae
        m['duels_won']                   = tk_w + ae_w

        # ── BALL RECOVERIES / POSITIONING (181-200) ───────────────────────────
        n_rec = len(recoveries)
        m['ball_recoveries']             = n_rec
        m['ball_recoveries_own_half']    = sum(1 for e in recoveries if sf(e.get('x'),50)<50)
        m['ball_recoveries_opp_half']    = sum(1 for e in recoveries if sf(e.get('x'),50)>=50)
        n_all = len(evts)
        m['total_actions']               = n_all
        m['actions_own_third']           = sum(1 for e in evts if in_own(sf(e.get('x'),50)))
        m['actions_mid_third']           = sum(1 for e in evts if in_mid(sf(e.get('x'),50)))
        m['actions_final_third']         = sum(1 for e in evts if in_ft(sf(e.get('x'),50)))
        m['actions_own_half']            = sum(1 for e in evts if sf(e.get('x'),50)<50)
        m['actions_opp_half']            = sum(1 for e in evts if sf(e.get('x'),50)>=50)
        xs = [sf(e.get('x'),50) for e in evts if e.get('x') is not None]
        ys = [sf(e.get('y'),50) for e in evts if e.get('y') is not None]
        m['avg_action_x']                = round(float(np.mean(xs)) if xs else 0,2)
        m['avg_action_y']                = round(float(np.mean(ys)) if ys else 0,2)
        m['actions_period1']             = sum(1 for e in evts if e.get('periodId')==1)
        m['actions_period2']             = sum(1 for e in evts if e.get('periodId')==2)
        mins = [e.get('timeMin',0) for e in evts if e.get('timeMin') is not None]
        m['first_action_minute']         = int(min(mins)) if mins else -1
        m['last_action_minute']          = int(max(mins)) if mins else -1
        m['minutes_active_proxy']        = int(max(mins)-min(mins)) if len(mins)>1 else 0
        m['actions_per_minute_active']   = round(n_all/max(m['minutes_active_proxy'],1),2)
        m['counterattack_involvements']  = sum(1 for e in evts if has_q(e,157))
        m['set_piece_involvements']      = sum(1 for e in evts if has_q(e,5) or has_q(e,107) or has_q(e,286))

        # ── SET PIECES (201-220) ──────────────────────────────────────────────
        m['corners_taken']               = sum(1 for e in evts if e.get('typeId')==CORNER_ID)
        m['corners_won']                 = sum(1 for e in evts if e.get('typeId')==CORNER_ID and e.get('outcome')==1)
        m['free_kicks_taken']            = fp
        m['direct_free_kicks']           = sum(1 for e in evts if e.get('typeId')==DFREE_ID)
        m['throw_ins_taken']             = ti
        m['throw_in_success_pct_sp']     = round(ti_succ/max(ti,1)*100,1)
        m['goal_kicks_taken']            = sum(1 for e in evts if e.get('typeId')==GK_KICK_ID)
        m['set_piece_shots']             = s_sp
        m['set_piece_goals']             = sum(1 for e in goals if has_q(e,5) or has_q(e,286) or has_q(e,9))
        m['goals_from_corners']          = sum(1 for e in goals if has_q(e,286))
        m['goals_from_free_kicks']       = sum(1 for e in goals if has_q(e,5))
        m['set_piece_key_passes']        = sum(1 for e in passes if (has_q(e,5) or has_q(e,286)) and has_q(e,210))
        m['free_kick_shots']             = sum(1 for e in shots  if has_q(e,5))
        m['set_piece_assists']           = sum(1 for e in evts   if (has_q(e,5) or has_q(e,286)) and has_q(e,189))
        m['penalty_won']                 = sum(1 for e in evts   if e.get('typeId')==PEN_ID and e.get('outcome')==1)
        m['penalty_scored']              = pen_g
        m['corners_into_box']            = sum(1 for e in evts if e.get('typeId')==CORNER_ID and sf(qval(e,140) or 0)>83)
        m['corners_short']               = sum(1 for e in evts if e.get('typeId')==CORNER_ID and sf(qval(e,140) or 0)<=83)
        m['set_piece_pass_total']        = sum(1 for e in passes if has_q(e,5) or has_q(e,107) or has_q(e,286))
        m['set_piece_pass_accuracy_pct'] = round(sum(1 for e in passes if (has_q(e,5) or has_q(e,107) or has_q(e,286)) and e.get('outcome')==1)/max(m['set_piece_pass_total'],1)*100,1)

        # ── GOALKEEPING (221-235) ─────────────────────────────────────────────
        n_saves = sum(1 for e in evts if e.get('typeId') in SAVE_IDS)
        shots_faced = sum(1 for e in evts if e.get('typeId') in ON_TGT)  # on-tgt faced = saves+goals conceded
        m['gk_saves']                    = n_saves
        m['gk_saves_period1']            = sum(1 for e in evts if e.get('typeId') in SAVE_IDS and e.get('periodId')==1)
        m['gk_saves_period2']            = sum(1 for e in evts if e.get('typeId') in SAVE_IDS and e.get('periodId')==2)
        m['gk_save_pct']                 = round(n_saves/max(shots_faced,1)*100,1)
        m['gk_claims']                   = sum(1 for e in evts if e.get('typeId')==CLAIM_ID)
        m['gk_punches']                  = sum(1 for e in evts if e.get('typeId')==CLAIM_ID and has_q(e,72))
        m['gk_throws']                   = sum(1 for e in evts if e.get('typeId')==GK_THROW_ID)
        m['gk_long_distribution']        = sum(1 for e in passes if has_q(e,211) or sf(qval(e,140) or 0)>50) if has_q(evts[0] if evts else {}, 34) else sum(1 for e in passes if sf(qval(e,140) or 0)>50 and pdist(sf(e.get('x'),50),sf(e.get('y'),50),sf(qval(e,140) or 50,50),sf(qval(e,141) or 50,50))>32)
        m['gk_short_distribution']       = sum(1 for e in passes if pdist(sf(e.get('x'),50),sf(e.get('y'),50),sf(qval(e,140) or 50,50),sf(qval(e,141) or 50,50))<20)
        m['gk_shots_faced']              = shots_faced
        m['gk_on_target_faced']          = shots_faced
        m['gk_distribution_total']       = sum(1 for e in evts if e.get('typeId') in (GK_THROW_ID,34,GK_KICK_ID))
        m['gk_distribution_accuracy']    = round(sum(1 for e in evts if e.get('typeId') in (GK_THROW_ID,34,GK_KICK_ID) and e.get('outcome')==1)/max(m['gk_distribution_total'],1)*100,1)
        m['gk_sweeper_actions']          = sum(1 for e in evts if e.get('typeId') in CLEAR_IDS and sf(e.get('x'),50)>83)

        # ── ADVANCED RATIOS (236-250) ─────────────────────────────────────────
        m['shot_on_target_per_pass']     = round(n_ot/max(n_p,1),4)
        m['dribble_to_key_pass_ratio']   = round(dr_succ/max(kp,1),3)
        m['aerial_contribution_index']   = round(ae_w*2 + n_ae,1)
        m['defensive_load_index']        = round(def_total / max(n_all,1) * 100, 2)
        m['attacking_contribution_index']= round((kp*2 + prog + n_sh*3 + n_g*5) / max(n_all,1) * 100, 2)
        m['shot_quality_index']          = round(m['xG_total']/max(n_sh,1) * m['shot_accuracy_pct'], 4)
        m['pass_progression_index']      = round((prog + into_ft + into_box) / max(n_p,1) * 100, 2)
        m['territorial_dominance_index'] = round(m['actions_opp_half'] / max(n_all,1) * 100, 2)
        m['pressing_contribution_pct']   = round(m['pressures_opp_half'] / max(n_all,1) * 100, 2)
        m['goal_involvement']            = n_g + kp  # goals + key passes as proxy for assists
        m['xG_involvement']              = round(m['xG_total'] + (sf(xg_p['xG'].sum() if not xg_p.empty else 0) * 0.5), 4)
        m['shots_on_target_per_shot']    = round(n_ot/max(n_sh,1),3)
        m['passes_per_action']           = round(n_p/max(n_all,1),3)
        m['duel_dominance_index']        = round((tk_w + ae_w) / max(n_tk + n_ae,1) * 100, 1)
        m['opp_half_action_pct']         = round(m['actions_opp_half']/max(n_all,1)*100,1)
        m['cross_accuracy_rate']         = round(m['crosses_successful']/max(m['crosses_total'],1)*100,1)
        m['tackle_interception_ratio']   = round(m['tackles_total']/max(m['interceptions_total'],1),3)
        m['shots_per_touch']             = round(n_sh/max(m['ball_touches'],1)*100,2)
        m['key_pass_per_touch']          = round(kp/max(m['ball_touches'],1)*100,2)
        m['counterattack_action_pct']    = round(m['counterattack_involvements']/max(n_all,1)*100,2)
        m['set_piece_involvement_pct']   = round(m['set_piece_involvements']/max(n_all,1)*100,2)

        return m, tm, side

    # ── build rows ────────────────────────────────────────────────────────────
    rows = []
    for pname, evts in player_events.items():
        if not evts: continue
        metrics, tm_name, side = player_metrics(pname, evts)
        row = {'match_id': os.path.splitext(fname)[0], 'date': date_str,
               'player': pname, 'team': tm_name, 'side': side}
        row.update(metrics)
        rows.append(row)

    return pd.DataFrame(rows)


def build_all(output_csv='all_player_metrics.csv'):
    files = sorted(set(
        glob.glob('/home/user/Project-Beth-Mead/WSL*/**/*.json', recursive=True) +
        glob.glob('/home/user/Project-Beth-Mead/WSL*/*.json')))
    print(f"Processing {len(files)} files...")
    frames = []
    for i, fp in enumerate(files):
        if i % 100 == 0: print(f"  {i}/{len(files)}")
        try:
            frames.append(extract_player_metrics(fp))
        except Exception as e:
            print(f"  SKIP {fp}: {e}")
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {output_csv}  ({len(df):,} rows × {len(df.columns)} cols)")
    return df


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--all':
        build_all()
    else:
        path = sys.argv[1] if len(sys.argv) > 1 else \
               sorted(glob.glob('/home/user/Project-Beth-Mead/WSL 2024-2025/DONE/*.json'))[0]
        print(f"Using: {path}\n")
        df = extract_player_metrics(path)
        mc = [c for c in df.columns if c not in ('match_id','date','player','team','side')]
        print(f"Metrics per player: {len(mc)}")
        print(f"Players extracted:  {len(df)}\n")
        pd.set_option('display.max_rows', 300)
        pd.set_option('display.max_columns', 10)
        pd.set_option('display.width', 120)
        # show a few players
        print(df.set_index('player')[mc].T.iloc[:, :4].to_string())
        out = os.path.splitext(path)[0] + '_player_metrics.csv'
        df.to_csv(out, index=False)
        print(f"\nSaved: {out}")
