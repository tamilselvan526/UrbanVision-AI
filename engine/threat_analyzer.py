"""
Algorithm 4 (Enhanced): Spatio-Temporal Threat & Action Reasoning Engine
=========================================================================
MAJOR UPGRADE — Now detects:

  NEW BEHAVIORS:
  ─────────────
  • STRIKING / HITTING: Rapid arm swing from pose motion velocity
  • STABBING / THRUSTING: Fast forward wrist thrust + elbow extension
  • CONFRONTATION / AGGRESSIVE STARE: Two people face-to-face at close range
  • IMPROVISED WEAPON IN HAND: Any object (bottle, pen, scissors, bat, etc.)
    within wrist proximity scores as potential weapon, boosted by context
  • FIGHTING: Two people in mutual close contact with striking motions
  • AGGRESSIVE APPROACH: Fast movement toward another person while making
    threatening gestures

  ENHANCED SCORING:
  ─────────────────
  • Wrist velocity → kinematic strike/stab score
  • Improvised weapon context penalty (elevated if person is also striking)
  • Confrontation radius detection (inter-person distance + face orientation)
  • Temporal escalation: score compounds across sustained threat windows
  • Victim scoring: detects fallen person near aggressor = assault in progress
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from engine.detector import Detection
from engine.tracker import TrackState
from engine.pose_estimator import PoseResult


# ─── Configuration ─────────────────────────────────────────────────────────

WEAPON_IN_HAND_DIST     = 90      # px: object within this dist of wrist = "held"
CONFRONTATION_DIST      = 180     # px: persons closer than this = potential confrontation
FACE_TO_FACE_DOT        = 0.5     # Dot product threshold for opposing face directions
THREAT_WINDOW_FRAMES    = 20      # Temporal smoothing window
AIM_ALIGNMENT_THRESH    = 0.72    # Aim vector → target alignment
STARE_DISTANCE_THRESH   = 200     # px: close face-to-face stare
FIGHTING_DIST_THRESH    = 130     # px: very close = active fight zone

# Improvised weapon context multiplier
# When a "borderline" object (bottle, pen) is detected near someone
# who is ALSO striking/in fighting stance, threat is escalated
IMPROVISED_WEAPON_CONTEXT_MULT = 1.6


# ─── Threat Levels ─────────────────────────────────────────────────────────

class ThreatLevel:
    NONE     = 0    # 0–20
    LOW      = 1    # 21–40   suspicious posture / approach
    MEDIUM   = 2    # 41–65   weapon present / confrontation
    HIGH     = 3    # 66–85   striking / stabbing / armed
    CRITICAL = 4    # 86–100  active assault / armed attack on person

THREAT_COLORS = {
    ThreatLevel.NONE:     (0, 160, 0),
    ThreatLevel.LOW:      (0, 160, 160),
    ThreatLevel.MEDIUM:   (0, 110, 220),
    ThreatLevel.HIGH:     (0, 40, 220),
    ThreatLevel.CRITICAL: (0, 0, 200),
}

THREAT_LABELS = {
    ThreatLevel.NONE:     "CLEAR",
    ThreatLevel.LOW:      "SUSPICIOUS",
    ThreatLevel.MEDIUM:   "THREAT DETECTED",
    ThreatLevel.HIGH:     "ACTIVE THREAT",
    ThreatLevel.CRITICAL: "CRITICAL THREAT",
}


# ─── Data Structures ──────────────────────────────────────────────────────

@dataclass
class ThreatEvent:
    timestamp:        float = field(default_factory=time.time)
    person_track_id:  int = -1
    weapon_class:     str = ""
    threat_score:     float = 0.0
    threat_level:     int = ThreatLevel.NONE
    threat_label:     str = "CLEAR"
    action_tags:      List[str] = field(default_factory=list)
    person_bbox:      Tuple[int, int, int, int] = (0, 0, 0, 0)
    person_center:    Tuple[int, int] = (0, 0)
    weapon_bbox:      Optional[Tuple[int, int, int, int]] = None
    weapon_center:    Optional[Tuple[int, int]] = None
    target_track_id:  Optional[int] = None
    snapshot:         Optional[np.ndarray] = None


@dataclass
class PersonThreatState:
    track_id:         int
    weapon_track_id:  Optional[int] = None
    weapon_class:     str = ""
    score_history:    deque = field(default_factory=lambda: deque(maxlen=THREAT_WINDOW_FRAMES))
    frames_armed:     int = 0
    frames_striking:  int = 0
    frames_confronting: int = 0
    last_event:       Optional[ThreatEvent] = None


# ─── Threat Analyzer ──────────────────────────────────────────────────────

class ThreatAnalyzer:
    """
    Algorithm 4 Enhanced — Spatio-Temporal Threat Reasoning Engine.

    Detects: weapon threats, strikes, stabs, confrontations, aggressive stares,
    improvised weapons (bottles/pens/scissors in hand), fighting, assaults.
    """

    def __init__(self, alert_threshold: float = 40.0):
        self.alert_threshold = alert_threshold
        self._person_states: Dict[int, PersonThreatState] = {}
        self.incident_log: List[ThreatEvent] = []
        logger.success("[ThreatAnalyzer] Algorithm 4 Enhanced (Behavioral AI) ready.")

    def analyze(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracks: Dict[int, TrackState],
        pose_results: List[PoseResult],
    ) -> List[ThreatEvent]:

        persons = [d for d in detections if d.is_person and d.track_id >= 0]
        weapons = [d for d in detections if d.is_weapon]

        # Build pose map by bbox IoU matching
        pose_map: Dict[int, PoseResult] = {}
        for pr in pose_results:
            tid = self._match_pose_to_person(pr, persons)
            if tid >= 0:
                pr.track_id = tid
                pose_map[tid] = pr

        # Associate weapons → persons (by wrist proximity)
        weapon_assignments: Dict[int, Detection] = {}
        for wpn in weapons:
            best_tid = self._assign_weapon_to_person(wpn, persons, pose_map)
            if best_tid >= 0:
                weapon_assignments[best_tid] = wpn

        # Score each person (suppress false alerts on display mannequins/dummies)
        frame_events: List[ThreatEvent] = []
        living_persons = [p for p in persons if not getattr(p, "is_mannequin", False)]
        for person in persons:
            if getattr(person, "is_mannequin", False):
                continue

            tid = person.track_id
            if tid not in self._person_states:
                self._person_states[tid] = PersonThreatState(track_id=tid)

            state  = self._person_states[tid]
            track  = tracks.get(tid)
            pose   = pose_map.get(tid)
            weapon = weapon_assignments.get(tid)

            score, tags = self._compute_threat_score(
                person=person, track=track, pose=pose,
                weapon=weapon, all_persons=living_persons,
                tracks=tracks, pose_map=pose_map,
            )

            # Update rolling state
            if weapon:
                state.weapon_track_id = weapon.track_id
                state.weapon_class    = weapon.class_name
                state.frames_armed   += 1
            else:
                state.frames_armed = max(0, state.frames_armed - 1)

            if pose and (pose.motion.is_striking or pose.motion.is_stabbing):
                state.frames_striking += 1
            else:
                state.frames_striking = max(0, state.frames_striking - 1)

            state.score_history.append(score)
            smooth = float(np.mean(state.score_history))
            level  = self._score_to_level(smooth)
            label  = THREAT_LABELS[level]

            target_tid = self._find_aim_target(person, pose, persons, tid) if pose else None

            event = ThreatEvent(
                person_track_id=tid,
                weapon_class=state.weapon_class,
                threat_score=smooth,
                threat_level=level,
                threat_label=label,
                action_tags=list(set(tags)),
                person_bbox=person.bbox,
                person_center=person.center,
                weapon_bbox=weapon.bbox if weapon else None,
                weapon_center=weapon.center if weapon else None,
                target_track_id=target_tid,
            )

            # Snapshot + log on new alert
            if smooth >= self.alert_threshold:
                if (state.last_event is None
                        or state.last_event.threat_level < level
                        or time.time() - state.last_event.timestamp > 4.0):
                    x1, y1, x2, y2 = person.bbox
                    h, w = frame.shape[:2]
                    pad = 40
                    crop = frame[max(0,y1-pad):min(h,y2+pad),
                                 max(0,x1-pad):min(w,x2+pad)]
                    event.snapshot = crop.copy() if crop.size > 0 else None
                    self.incident_log.append(event)
                    logger.warning(
                        f"[ALERT] Person #{tid} | {label} | "
                        f"Score={smooth:.0f}% | Tags={list(set(tags))}"
                    )

            state.last_event = event
            frame_events.append(event)

        # Prune stale states
        active = {p.track_id for p in persons}
        for tid in [t for t in self._person_states if t not in active]:
            del self._person_states[tid]

        return frame_events

    # ─── Core Scoring Engine ─────────────────────────────────────────────

    def _compute_threat_score(
        self,
        person: Detection,
        track: Optional[TrackState],
        pose: Optional[PoseResult],
        weapon: Optional[Detection],
        all_persons: List[Detection],
        tracks: Dict[int, TrackState],
        pose_map: Dict[int, PoseResult],
    ) -> Tuple[float, List[str]]:

        score = 0.0
        tags: List[str] = []

        # ── Feature 1: Detected weapon / object ─────────────────────────
        if weapon:
            wpn = weapon.class_name.lower()
            score += 20.0
            tags.append("OBJECT_DETECTED")

            # Classify severity by weapon type
            if any(w in wpn for w in ["gun","pistol","rifle","firearm","revolver","shotgun"]):
                score += 25.0
                tags.append("FIREARM")
            elif any(w in wpn for w in ["knife","blade","machete","sword"]):
                score += 18.0
                tags.append("BLADE")
            elif any(w in wpn for w in ["baseball bat","bat","hammer","pipe","rod"]):
                score += 12.0
                tags.append("BLUNT_WEAPON")
            elif any(w in wpn for w in ["scissors","fork","bottle","glass"]):
                score += 8.0
                tags.append("IMPROVISED_WEAPON")
            else:
                score += 5.0
                tags.append("POTENTIAL_WEAPON")

        # ── Feature 2: Object in hand (wrist proximity) ─────────────────
        weapon_in_hand = False
        if weapon and pose:
            for wrist in [pose.left_wrist, pose.right_wrist]:
                if wrist:
                    d = math.dist(wrist, weapon.center)
                    if d < WEAPON_IN_HAND_DIST:
                        weapon_in_hand = True
                        break
            if weapon_in_hand:
                score += 18.0
                tags.append("WEAPON_IN_HAND")

                # Context multiplier: improvised weapon + aggressive action = escalate
                is_improvised = any(t in tags for t in [
                    "IMPROVISED_WEAPON", "POTENTIAL_WEAPON", "BLUNT_WEAPON"
                ])
                if is_improvised and pose and (pose.motion.is_striking or pose.motion.is_stabbing):
                    score *= IMPROVISED_WEAPON_CONTEXT_MULT
                    tags.append("IMPROVISED_WEAPON_ACTIVE")

        elif weapon:
            d = math.dist(person.center, weapon.center)
            if d < 120:
                score += 8.0
                tags.append("WEAPON_NEAR_PERSON")

        # ── Feature 3: STRIKING motion (fast arm swing) ──────────────────
        if pose and pose.motion.is_striking:
            vel = pose.motion.max_wrist_vel
            strike_score = min(vel * 1.2, 22.0)   # scale with velocity, cap at 22
            score += strike_score
            tags.append(f"STRIKING ({pose.motion.strike_arm.upper()} ARM)")
            state = self._person_states.get(person.track_id)
            if state and state.frames_striking > 5:
                score += 8.0
                tags.append("SUSTAINED_ATTACK")

        # ── Feature 4: STABBING motion (fast thrust + elbow extension) ───
        if pose and pose.motion.is_stabbing:
            vel = pose.motion.max_wrist_vel
            stab_score = min(vel * 1.5, 28.0)
            score += stab_score
            tags.append(f"STABBING_MOTION ({pose.motion.stab_arm.upper()} ARM)")

        # ── Feature 5: Posture assessment ───────────────────────────────
        if pose:
            posture_scores = {
                "stabbing":       0.0,  # already scored above
                "striking":       0.0,  # already scored above
                "aiming":         16.0,
                "fighting_stance":12.0,
                "arms_raised":    7.0,
                "crouching":      4.0,
                "fallen":         2.0,
                "standing":       0.0,
                "unknown":        0.0,
            }
            ps = posture_scores.get(pose.posture, 0.0)
            if ps > 0:
                score += ps
                tags.append(f"POSTURE_{pose.posture.upper()}")

        # ── Feature 6: CONFRONTATION / ACTIVE FIGHTING / AGGRESSIVE STARE ────
        confrontation_target = None
        person_h = abs(person.bbox[3] - person.bbox[1])

        for other in all_persons:
            if other.track_id == person.track_id:
                continue
            dist_to_other = math.dist(person.center, other.center)
            other_h = abs(other.bbox[3] - other.bbox[1])
            max_h = max(person_h, other_h, 100)

            # Dynamic scale based on body size (works for 1080p, 4K, and 480p)
            dyn_confront_dist = max(CONFRONTATION_DIST, int(max_h * 1.6))
            dyn_fight_dist = max(FIGHTING_DIST_THRESH, int(max_h * 1.1))

            if dist_to_other < dyn_confront_dist:
                score += 8.0
                tags.append("CLOSE_PROXIMITY")

                other_pose = pose_map.get(other.track_id)

                # 1. DUAL COMBAT STANCE / ACTIVE FIGHTING ENGAGEMENT
                # Both subjects squaring up facing each other in fighting/aiming stance
                if pose and other_pose:
                    my_combat = pose.posture in ("fighting_stance", "aiming", "striking") or "AIMING_AT_PERSON" in tags
                    their_combat = other_pose.posture in ("fighting_stance", "aiming", "striking")

                    if my_combat and their_combat:
                        score += 35.0
                        tags.append("ACTIVE_FIGHTING_ENGAGEMENT")
                        tags.append("DUAL_COMBAT_STANCE")

                        # If strikes are thrown or rapid wrist movement
                        if pose.motion.is_striking or other_pose.motion.is_striking:
                            score += 20.0
                            tags.append("STRIKES_EXCHANGED")

                        # Aggressive combat lunge
                        if track and (track.sudden_move or track.speed > 20):
                            score += 12.0
                            tags.append("COMBAT_LUNGE")

                # 2. Face-to-face aggressive stare detection
                if (pose and pose.face_direction and other_pose
                        and other_pose.face_direction and
                        dist_to_other < max(STARE_DISTANCE_THRESH, int(max_h * 1.2))):
                    dot = (pose.face_direction[0] * (-other_pose.face_direction[0]) +
                           pose.face_direction[1] * (-other_pose.face_direction[1]))
                    if dot > FACE_TO_FACE_DOT:
                        score += 12.0
                        tags.append("CONFRONTATION_STARE")
                        confrontation_target = other.track_id

                # 3. Active fight zone (striking within close body range)
                if dist_to_other < dyn_fight_dist:
                    if pose and pose.motion.is_striking:
                        score += 20.0
                        tags.append("ACTIVE_FIGHT")
                    if (other_pose and other_pose.motion.is_striking
                            and pose and pose.motion.is_striking):
                        score += 15.0
                        tags.append("MUTUAL_COMBAT")

        # ── Feature 7: Victim nearby (fallen person) ────────────────────
        for other in all_persons:
            if other.track_id == person.track_id:
                continue
            other_pose = pose_map.get(other.track_id)
            if other_pose and other_pose.posture == "fallen":
                dist = math.dist(person.center, other.center)
                if dist < CONFRONTATION_DIST * 1.5:
                    score += 12.0
                    tags.append("VICTIM_DOWN_NEARBY")
                    break

        # ── Feature 8: Aim vector at another person ──────────────────────
        if pose and all_persons:
            for target in all_persons:
                if target.track_id == person.track_id:
                    continue
                alignment = self._aim_alignment(person.center, pose, target.center)
                if alignment > AIM_ALIGNMENT_THRESH:
                    score += 14.0
                    tags.append("AIMING_AT_PERSON")
                    break

        # ── Feature 9: Kinematic movement features ───────────────────────
        if track:
            if track.sudden_move:
                score += 5.0
                tags.append("SUDDEN_MOVEMENT")
            if track.speed > 40:
                score += 3.0
                tags.append("RUNNING")
            # Accelerating toward another person
            for other in all_persons:
                if other.track_id == person.track_id:
                    continue
                ot = tracks.get(other.track_id)
                if ot and self._approaching(track, ot):
                    score += 7.0
                    tags.append("CHARGING_TOWARD_PERSON")
                    break

        # ── Feature 10: Sustained threat bonus ──────────────────────────
        state = self._person_states.get(person.track_id)
        if state:
            if state.frames_armed > 25:
                score += 4.0
                tags.append("SUSTAINED_ARMED")
            if state.frames_striking > 8:
                score += 5.0
                tags.append("SUSTAINED_ASSAULT")

        return min(score, 100.0), tags

    # ─── Helpers ─────────────────────────────────────────────────────────

    def _aim_alignment(self, shooter: Tuple, pose: PoseResult, target: Tuple) -> float:
        dx, dy = target[0]-shooter[0], target[1]-shooter[1]
        mag = math.sqrt(dx**2+dy**2)
        if mag < 1e-6: return 0.0
        tv = (dx/mag, dy/mag)
        best = 0.0
        for v in [pose.aim_vector_left, pose.aim_vector_right]:
            if v:
                best = max(best, v[0]*tv[0]+v[1]*tv[1])
        return best

    def _approaching(self, mover: TrackState, target: TrackState) -> bool:
        if not mover.trail or not target.trail: return False
        mx, my = mover.trail[-1]
        tx, ty = target.trail[-1]
        dx, dy = tx-mx, ty-my
        mag = math.sqrt(dx**2+dy**2)
        if mag < 1e-6: return False
        vx, vy = mover.velocity
        return (vx*dx+vy*dy)/mag > 5.0

    def _match_pose_to_person(self, pose: PoseResult, persons: List[Detection]) -> int:
        best_tid, best_iou = -1, 0.25
        px1, py1, px2, py2 = pose.person_bbox
        for p in persons:
            bx1, by1, bx2, by2 = p.bbox
            ix1, iy1 = max(px1,bx1), max(py1,by1)
            ix2, iy2 = min(px2,bx2), min(py2,by2)
            if ix2<=ix1 or iy2<=iy1: continue
            inter = (ix2-ix1)*(iy2-iy1)
            union = (px2-px1)*(py2-py1)+(bx2-bx1)*(by2-by1)-inter
            iou = inter/(union+1e-6)
            if iou > best_iou:
                best_iou, best_tid = iou, p.track_id
        return best_tid

    def _assign_weapon_to_person(
        self, weapon: Detection, persons: List[Detection],
        pose_map: Dict[int, PoseResult]
    ) -> int:
        best_tid, best_dist = -1, 220.0
        wx, wy = weapon.center
        for p in persons:
            pose = pose_map.get(p.track_id)
            if pose:
                for wrist in [pose.left_wrist, pose.right_wrist]:
                    if wrist:
                        d = math.dist((wx,wy), wrist)
                        if d < best_dist:
                            best_dist, best_tid = d, p.track_id
            else:
                d = math.dist((wx,wy), p.center)
                if d < best_dist:
                    best_dist, best_tid = d, p.track_id
        return best_tid

    def _find_aim_target(
        self, shooter: Detection, pose: Optional[PoseResult],
        all_persons: List[Detection], shooter_tid: int
    ) -> Optional[int]:
        if not pose: return None
        best_tid, best_align = None, AIM_ALIGNMENT_THRESH
        for t in all_persons:
            if t.track_id == shooter_tid: continue
            a = self._aim_alignment(shooter.center, pose, t.center)
            if a > best_align:
                best_align, best_tid = a, t.track_id
        return best_tid

    @staticmethod
    def _score_to_level(score: float) -> int:
        if score <= 20:   return ThreatLevel.NONE
        if score <= 40:   return ThreatLevel.LOW
        if score <= 65:   return ThreatLevel.MEDIUM
        if score <= 85:   return ThreatLevel.HIGH
        return ThreatLevel.CRITICAL

    # ─── Visualization ────────────────────────────────────────────────────

    def draw_threat_overlay(self, frame: np.ndarray, events: List[ThreatEvent]) -> np.ndarray:
        vis = frame.copy()
        h, w = vis.shape[:2]

        for event in events:
            color = THREAT_COLORS[event.threat_level]
            x1, y1, x2, y2 = event.person_bbox

            # Person border (thickness scales with threat)
            thick = max(1, event.threat_level + 1)
            cv2.rectangle(vis, (x1,y1), (x2,y2), color, thick)

            # Weapon → wrist link
            if event.weapon_center:
                cv2.line(vis, event.person_center, event.weapon_center, (0,30,200), 2, cv2.LINE_AA)
                cv2.circle(vis, event.weapon_center, 9, (0,0,180), -1)
                cv2.putText(vis, event.weapon_class.upper(),
                            (event.weapon_center[0]+10, event.weapon_center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0,0,200), 1, cv2.LINE_AA)

            # Score bar
            bar_w  = x2-x1
            fill_w = int(bar_w * event.threat_score / 100.0)
            cv2.rectangle(vis, (x1, y2+2), (x2, y2+8), (35,35,35), -1)
            cv2.rectangle(vis, (x1, y2+2), (x1+fill_w, y2+8), color, -1)

            # Label badge
            label = f"{event.threat_label} [{event.threat_score:.0f}%]"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(vis, (x1, y1-th-10), (x1+tw+6, y1), color, -1)
            cv2.putText(vis, label, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1, cv2.LINE_AA)

            # Action tags (right side, max 5)
            for k, tag in enumerate(event.action_tags[:5]):
                cv2.putText(vis, f"> {tag}", (x2+5, y1+14+k*14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, color, 1, cv2.LINE_AA)

        # Global HUD
        if events:
            ms  = max(e.threat_score for e in events)
            ml  = max(e.threat_level for e in events)
            hc  = THREAT_COLORS[ml]
            hl  = THREAT_LABELS[ml]
            cv2.rectangle(vis, (w-225,10), (w-10,95), (18,18,18), -1)
            cv2.rectangle(vis, (w-225,10), (w-10,95), hc, 2)
            cv2.putText(vis, "THREAT LEVEL", (w-220,28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
            cv2.putText(vis, hl, (w-220,52),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, hc, 2)
            bf = int(195 * ms/100.0)
            cv2.rectangle(vis, (w-220,60), (w-25,75), (45,45,45), -1)
            cv2.rectangle(vis, (w-220,60), (w-220+bf,75), hc, -1)
            cv2.putText(vis, f"{ms:.0f}%", (w-50,73),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255,255,255), 1)

        return vis
