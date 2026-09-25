"""
Phase 1 starter: run pose estimation on a volleyball video.

Usage:
    python pose_check.py "volleyball hitting side view.mov" right

It creates:
    <name>_annotated.mp4   the video with the hitter's skeleton drawn on it
    <name>_angles.csv      one row per frame with:
        elbow_angle, knee_angle, shoulder_angle   degrees, hitting side
        trunk_lean      degrees, 0 = upright, + = leaning forward, - = leaning back
        hip_height      hitting hip, above the bottom of the video, in torso lengths
        wrist_height    hitting wrist, above the bottom of the video, in torso lengths
        wrist_above_head          hitting wrist above the nose, in torso lengths
        wrist_ahead_of_shoulder   + = wrist in front of the shoulder, in torso lengths
        torso_px        torso length in pixels (the scale used for that frame)

The video can have other people in it (a setter, people walking by).
The script finds everyone, follows each person from frame to frame,
and treats the person who moves the most as the hitter.
"""

import csv
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# MediaPipe 1.x removed the old "mp.solutions.pose" API.
# The new API (PoseLandmarker) needs a model file. We use the "heavy" model,
# which is the most accurate one (same idea as model_complexity=2 before).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)
MODEL_PATH = Path(__file__).parent / "models" / "pose_landmarker_heavy.task"

L = vision.PoseLandmark  # names for the 33 body points, like L.RIGHT_ELBOW
POSE_CONNECTIONS = vision.PoseLandmarksConnections.POSE_LANDMARKS  # skeleton lines

MIN_VISIBILITY = 0.5  # ignore points the AI is not confident about
MAX_PEOPLE = 3  # how many people MediaPipe looks for in each frame

# Settings for following people from frame to frame (see build_tracks)
MAX_STEP_PER_FRAME = 0.6  # how far a person can move in one frame, in torso lengths
MAX_MISSING_FRAMES = 10  # how long a person can disappear and still be the same person


def download_model():
    """Download the pose model the first time the script runs."""
    if MODEL_PATH.exists():
        return
    MODEL_PATH.parent.mkdir(exist_ok=True)
    print(f"Downloading pose model to {MODEL_PATH} ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


def angle(a, b, c):
    """Angle at point b (in degrees) formed by the points a-b-c."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def get_point(landmarks, index, width, height):
    """Return (x, y) in pixels, or None if the AI is not sure about it."""
    p = landmarks[index]
    if p.visibility < MIN_VISIBILITY:
        return None
    return (p.x * width, p.y * height)


def joint_angle(landmarks, idx_a, idx_b, idx_c, width, height):
    pts = [get_point(landmarks, i, width, height) for i in (idx_a, idx_b, idx_c)]
    if any(p is None for p in pts):
        return None
    return angle(*pts)


def draw_skeleton(frame, landmarks, width, height):
    """Draw the body points and the lines between them.

    The new API has no simple drawing helper for this, so we use OpenCV.
    Points the AI is not sure about are skipped.
    """
    for conn in POSE_CONNECTIONS:
        a = get_point(landmarks, conn.start, width, height)
        b = get_point(landmarks, conn.end, width, height)
        if a is not None and b is not None:
            cv2.line(frame, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), (255, 255, 255), 2)
    for i in range(len(landmarks)):
        p = get_point(landmarks, i, width, height)
        if p is not None:
            cv2.circle(frame, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1)


def body_center_and_size(landmarks, width, height):
    """Return the middle of the hips (pixels) and the torso length (pixels).

    We use every point here, even low-visibility ones, because we only need
    a rough position to tell people apart, not an exact measurement.
    Torso length = distance from the middle of the shoulders to the middle of the hips.
    """
    def px(i):
        return np.array([landmarks[i].x * width, landmarks[i].y * height])

    hips = (px(L.LEFT_HIP) + px(L.RIGHT_HIP)) / 2
    shoulders = (px(L.LEFT_SHOULDER) + px(L.RIGHT_SHOULDER)) / 2
    torso = max(float(np.linalg.norm(shoulders - hips)), 1.0)
    return hips, torso


def detect_people(video_path, fps):
    """Pass 1: run MediaPipe on every frame.

    Returns a list with one entry per frame. Each entry is a list of people,
    and each person is a list of 33 landmarks.
    """
    download_model()
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO,  # uses earlier frames to track better
        num_poses=MAX_PEOPLE,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    people_per_frame = []
    cap = cv2.VideoCapture(str(video_path))
    frame_number = 0
    with vision.PoseLandmarker.create_from_options(options) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # MediaPipe wants RGB, OpenCV gives BGR
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            # VIDEO mode needs the time of each frame in whole milliseconds
            timestamp_ms = int(frame_number * 1000 / fps)
            result = pose.detect_for_video(mp_image, timestamp_ms)
            people_per_frame.append(result.pose_landmarks)
            frame_number += 1
    cap.release()
    return people_per_frame


def build_tracks(people_per_frame, width, height):
    """Follow each person from frame to frame.

    A "track" is one person over time. For each frame, every person found is
    added to the track whose last known hip position is closest, as long as it
    is close enough. If no track is close, that person starts a new track.

    Each track is a dict: {frame_number: landmarks, ...}
    """
    tracks = []  # list of dicts {frame: landmarks}
    last_seen = []  # for each track: (frame, hip_center, torso_length)

    for frame_number, people in enumerate(people_per_frame):
        used_tracks = set()  # two people in one frame cannot join the same track
        for landmarks in people:
            center, torso = body_center_and_size(landmarks, width, height)

            best_track, best_dist = None, None
            for t, (seen_frame, seen_center, seen_torso) in enumerate(last_seen):
                gap = frame_number - seen_frame
                if t in used_tracks or gap > MAX_MISSING_FRAMES:
                    continue
                dist = np.linalg.norm(center - seen_center)
                # The longer someone was missing, the further they may have moved
                limit = MAX_STEP_PER_FRAME * seen_torso * gap
                if dist <= limit and (best_dist is None or dist < best_dist):
                    best_track, best_dist = t, dist

            if best_track is None:
                tracks.append({})
                last_seen.append(None)
                best_track = len(tracks) - 1

            tracks[best_track][frame_number] = landmarks
            last_seen[best_track] = (frame_number, center, torso)
            used_tracks.add(best_track)

    return tracks


def track_movement(track, width, height):
    """Total distance the hips travel, measured in torso lengths.

    Using torso lengths means a person close to the camera (who looks big)
    does not count as moving more than a person far away.
    """
    total = 0.0
    previous = None
    for frame_number in sorted(track):
        center, torso = body_center_and_size(track[frame_number], width, height)
        if previous is not None:
            total += np.linalg.norm(center - previous) / torso
        previous = center
    return total


def smoothed_torso_lengths(track, width, height, window=7):
    """Torso length (pixels) for every frame of a track, smoothed over time.

    We divide heights by torso length so that camera distance does not matter.
    A single blurry frame can give a wrong torso length, so for each frame we
    take the median of the frames around it (7 frames = about 0.25 s).
    """
    frames = sorted(track)
    raw = {f: body_center_and_size(track[f], width, height)[1] for f in frames}
    smooth = {}
    for f in frames:
        nearby = [raw[g] for g in frames if abs(g - f) <= window // 2]
        smooth[f] = float(np.median(nearby))
    return smooth


def facing_direction(track):
    """Return +1 if the hitter faces the right side of the video, -1 if the left.

    In a side view the nose is in front of the ears, so we check which side of
    the ears the nose is on, and take the most common answer over all frames.
    """
    votes = [
        lm[L.NOSE].x - (lm[L.LEFT_EAR].x + lm[L.RIGHT_EAR].x) / 2
        for lm in track.values()
    ]
    return 1 if np.median(votes) > 0 else -1


def trunk_lean(landmarks, shoulder_idx, hip_idx, facing, width, height):
    """How far the trunk tilts from straight up, in degrees.

    0 = upright, positive = leaning forward (toward where the hitter faces),
    negative = leaning back. Uses the line from the hip to the shoulder.
    """
    s = get_point(landmarks, shoulder_idx, width, height)
    h = get_point(landmarks, hip_idx, width, height)
    if s is None or h is None:
        return None
    forward = (s[0] - h[0]) * facing  # how far the shoulder is ahead of the hip
    up = h[1] - s[1]  # image y grows downward, so this is "shoulder above hip"
    return float(np.degrees(np.arctan2(forward, up)))


def height_in_torsos(landmarks, index, torso_px, width, height):
    """Height of a point above the bottom of the video, in torso lengths.

    This is not the height above the floor. Use it to compare frames that are
    close in time (for example: hips at the plant vs hips at the top of the jump).
    """
    p = get_point(landmarks, index, width, height)
    if p is None:
        return None
    return (height - p[1]) / torso_px


def wrist_above_head(landmarks, wrist_idx, torso_px, width, height):
    """How far the wrist is above the head, in torso lengths.

    MediaPipe has no "top of the head" point, so we use the nose as the head.
    Positive = wrist above the nose. Both points come from the same frame,
    so camera distance and camera tilt do not matter here.
    """
    w = get_point(landmarks, wrist_idx, width, height)
    n = get_point(landmarks, L.NOSE, width, height)
    if w is None or n is None:
        return None
    return (n[1] - w[1]) / torso_px  # image y grows downward


def wrist_ahead_of_shoulder(landmarks, wrist_idx, shoulder_idx, facing, torso_px, width, height):
    """How far the wrist is in front of the shoulder, in torso lengths.

    Positive = in front (toward where the hitter faces), negative = behind.
    """
    w = get_point(landmarks, wrist_idx, width, height)
    s = get_point(landmarks, shoulder_idx, width, height)
    if w is None or s is None:
        return None
    return (w[0] - s[0]) * facing / torso_px


def main():
    if len(sys.argv) < 3 or sys.argv[2] not in ("left", "right"):
        print("Usage: python pose_check.py <video> <left|right>")
        sys.exit(1)

    video_path = Path(sys.argv[1])
    side = sys.argv[2]

    # Pick the landmarks for the hitting arm and the same-side leg
    if side == "right":
        shoulder, elbow, wrist = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST
        hip, knee, ankle = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE
    else:
        shoulder, elbow, wrist = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
        hip, knee, ankle = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open {video_path}")
        sys.exit(1)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Pass 1: find everyone in every frame
    people_per_frame = detect_people(video_path, fps)

    # Follow each person over time, then pick the one who moves the most
    tracks = build_tracks(people_per_frame, width, height)
    if not tracks:
        print("No people found in the video.")
        sys.exit(1)
    movements = [track_movement(t, width, height) for t in tracks]
    hitter_index = int(np.argmax(movements))
    hitter = tracks[hitter_index]

    print(f"People tracks found: {len(tracks)}")
    for i, (t, m) in enumerate(zip(tracks, movements)):
        first, last = min(t), max(t)
        mark = "  <- hitter" if i == hitter_index else ""
        print(f"  track {i}: frames {first}-{last}, moved {m:.1f} torso lengths{mark}")

    torso_lengths = smoothed_torso_lengths(hitter, width, height)
    facing = facing_direction(hitter)
    print(f"Hitter faces the {'right' if facing > 0 else 'left'} side of the video")

    # Pass 2: read the video again, draw the hitter, and save the angles
    out_video = video_path.with_name(video_path.stem + "_annotated.mp4")
    out_csv = video_path.with_name(video_path.stem + "_angles.csv")
    writer = cv2.VideoWriter(
        str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )

    rows = []
    cap = cv2.VideoCapture(str(video_path))
    for frame_number in range(len(people_per_frame)):
        ok, frame = cap.read()
        if not ok:
            break

        # Every value starts as None ("not measured") and is filled in if we can
        elbow_angle = knee_angle = shoulder_angle = lean = None
        hip_height = wrist_height = torso_px = None
        above_head = ahead = None

        lm = hitter.get(frame_number)  # None if the hitter was not found in this frame
        if lm is not None:
            torso_px = torso_lengths[frame_number]
            elbow_angle = joint_angle(lm, shoulder, elbow, wrist, width, height)
            knee_angle = joint_angle(lm, hip, knee, ankle, width, height)
            # Shoulder angle: 0 = arm down by the side, 180 = arm straight up
            shoulder_angle = joint_angle(lm, hip, shoulder, elbow, width, height)
            lean = trunk_lean(lm, shoulder, hip, facing, width, height)
            hip_height = height_in_torsos(lm, hip, torso_px, width, height)
            wrist_height = height_in_torsos(lm, wrist, torso_px, width, height)
            above_head = wrist_above_head(lm, wrist, torso_px, width, height)
            ahead = wrist_ahead_of_shoulder(lm, wrist, shoulder, facing, torso_px, width, height)
            draw_skeleton(frame, lm, width, height)

        # Write the angles on the video so you can check them by eye
        labels = [
            ("elbow", elbow_angle),
            ("knee", knee_angle),
            ("shoulder", shoulder_angle),
            ("lean", lean),
        ]
        for i, (name, value) in enumerate(labels):
            text = f"{name}: {value:.0f}" if value is not None else f"{name}: --"
            cv2.putText(frame, text, (20, 40 + 35 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        writer.write(frame)

        def rounded(value, digits):
            """Round a number for the CSV, or leave the cell empty if it is None."""
            return "" if value is None else round(value, digits)

        rows.append(
            {
                "frame": frame_number,
                "time_sec": round(frame_number / fps, 3),
                "elbow_angle": rounded(elbow_angle, 1),
                "knee_angle": rounded(knee_angle, 1),
                "shoulder_angle": rounded(shoulder_angle, 1),
                "trunk_lean": rounded(lean, 1),
                "hip_height": rounded(hip_height, 3),  # in torso lengths
                "wrist_height": rounded(wrist_height, 3),  # in torso lengths
                "wrist_above_head": rounded(above_head, 3),  # in torso lengths
                "wrist_ahead_of_shoulder": rounded(ahead, 3),  # in torso lengths
                "torso_px": rounded(torso_px, 1),  # the scale used for this frame
            }
        )

    cap.release()
    writer.release()

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    tracked = sum(1 for r in rows if r["elbow_angle"] != "")
    print(f"Frames: {len(rows)}, elbow tracked in {tracked} ({100 * tracked / max(len(rows), 1):.0f}%)")
    print(f"Saved: {out_video}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
