"""
Find the contact frame and the jump of a spike, and print the key numbers.

Run pose_check.py first. It makes the CSV this script reads.

Usage:
    python find_contact.py "volleyball hitting side view_angles.csv"

How contact is found:
    The ball is not tracked yet, so we guess that contact happens when the
    hitting wrist is at its highest point during the jump.

    1. Find the jump: the frame where the hips have risen the most compared
       to their lowest point in the second before.
    2. Contact = the frame with the highest wrist near that jump
       (with a small tie-break, see WRIST_TIE_MARGIN).

How the jump is measured:
    Plant = the lowest hip point just before takeoff (see find_plant).
    Jump height = how far the hips rise from the plant to the top of the jump.

    Why not just "highest wrist in the whole video"? If the player walks away
    from the camera, they appear higher in the picture, so a frame where they
    are just walking far away can look "higher" than the real spike.
    Comparing over a short time window avoids that.
"""

import csv
import sys

JUMP_LOOKBACK_SEC = 1.0  # compare the hips to their lowest point in this much time before
CONTACT_SEARCH_SEC = 0.4  # look for the highest wrist this close to the jump peak
# If two frames have almost the same wrist height (within this many torso
# lengths), they count as a tie, and we pick the one with the straighter elbow.
# Keep this small: we are choosing the frame partly by elbow angle, which is
# also something we measure, so a big margin could hide a real bent-arm hit.
WRIST_TIE_MARGIN = 0.05
# Hip heights closer than this (in torso lengths) count as "the same" around
# the plant. It is used two ways (see find_plant):
#   - small bumps smaller than this are noise, not the end of the plant
#   - frames this close to the lowest hip are a tie, and the most bent knee wins
# Same warning as WRIST_TIE_MARGIN: keep it small, because the tie-break picks
# the frame partly by knee angle, which is also something we measure.
PLANT_TOLERANCE = 0.05
NEIGHBOR_FRAMES = 2  # also show this many frames before and after contact
TRACKING_WINDOW = 15  # check tracking this many frames around contact (about 0.5 s)


def load_rows(csv_path):
    """Read the CSV. Numbers become floats, empty cells become None."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            clean = {}
            for key, value in row.items():
                clean[key] = float(value) if value != "" else None
            clean["frame"] = int(clean["frame"])
            rows.append(clean)
    return rows


def find_jump_peak(rows):
    """Return the row where the hips rose the most in a short time, or None.

    For every frame we look back JUMP_LOOKBACK_SEC seconds, find the lowest
    hip height in that time, and measure how much higher the hips are now.
    The frame with the biggest rise is the top of the jump.
    """
    best_row, best_rise = None, None
    for r in rows:
        if r["hip_height"] is None:
            continue
        before = [
            b["hip_height"]
            for b in rows
            if b["hip_height"] is not None
            and r["time_sec"] - JUMP_LOOKBACK_SEC <= b["time_sec"] < r["time_sec"]
        ]
        if not before:
            continue
        rise = r["hip_height"] - min(before)
        if best_rise is None or rise > best_rise:
            best_row, best_rise = r, rise
    return best_row


def find_contact(rows, jump_row):
    """Find the contact frame near the jump peak.

    Returns (contact_row, highest_row). They are the same row unless the
    tie-break picked a different frame. Returns (None, None) if the wrist
    was not tracked near the jump.
    """
    nearby = [
        r
        for r in rows
        if r["wrist_height"] is not None
        and abs(r["time_sec"] - jump_row["time_sec"]) <= CONTACT_SEARCH_SEC
    ]
    if not nearby:
        return None, None
    highest = max(nearby, key=lambda r: r["wrist_height"])

    # Tie-break: frames almost as high as the highest one, with a tracked elbow
    ties = [
        r
        for r in nearby
        if r["wrist_height"] >= highest["wrist_height"] - WRIST_TIE_MARGIN
        and r["elbow_angle"] is not None
    ]
    if not ties:
        return highest, highest
    straightest = max(ties, key=lambda r: r["elbow_angle"])
    return straightest, highest


def hip_pixels(row):
    """Hip height above the bottom of the video, in pixels.

    The CSV stores hip height in torso lengths, so multiply by the torso
    length (pixels) for that frame to get pixels back.
    """
    return row["hip_height"] * row["torso_px"]


def find_plant(rows, jump_row):
    """Find the plant: the lowest hip point just before takeoff.

    Step 1: start at the jump and walk back in time. While the hips keep
    getting lower, keep going. When they are clearly higher again, we have
    passed the plant and stop.

    Step 2 (tie-break): around the plant the hips barely move for a few
    frames, but the knee angle changes a lot. So among the frames we walked
    through whose hips are almost as low as the lowest, pick the most bent knee.

    Hip heights are compared in pixels (divided by the torso length at the
    lowest frame), so frames with slightly different torso sizes are fair.

    Why not "lowest hips in the whole second before the jump"? In this kind of
    video the player can be closer to the camera during the run-in, which makes
    them look lower in the picture even though they are not.

    Returns (plant_row, lowest_row). They are the same row unless the
    tie-break picked a different frame. Returns (None, None) if no hips.
    """
    before = [
        r
        for r in rows
        if r["hip_height"] is not None
        and jump_row["time_sec"] - JUMP_LOOKBACK_SEC <= r["time_sec"] <= jump_row["time_sec"]
    ]

    # Step 1: walk back in time and remember every frame we pass
    lowest = None
    walked = []
    for r in reversed(before):  # newest frame first, going back in time
        if lowest is not None:
            above_lowest = (hip_pixels(r) - hip_pixels(lowest)) / lowest["torso_px"]
            if above_lowest > PLANT_TOLERANCE:
                break  # hips are clearly higher again: we went past the plant
        walked.append(r)
        if lowest is None or hip_pixels(r) <= hip_pixels(lowest):
            lowest = r
    if lowest is None:
        return None, None

    # Step 2: tie-break by the most bent knee (smallest knee angle)
    limit_px = hip_pixels(lowest) + PLANT_TOLERANCE * lowest["torso_px"]
    ties = [r for r in walked if hip_pixels(r) <= limit_px and r["knee_angle"] is not None]
    if not ties:
        return lowest, lowest
    most_bent = min(ties, key=lambda r: r["knee_angle"])
    return most_bent, lowest


def find_top_of_jump(rows, plant_row):
    """Return the row with the highest hips in the second after the plant."""
    after = [
        r
        for r in rows
        if r["hip_height"] is not None
        and plant_row["time_sec"] < r["time_sec"] <= plant_row["time_sec"] + JUMP_LOOKBACK_SEC
    ]
    if not after:
        return None
    return max(after, key=hip_pixels)


def jump_height(plant_row, top_row):
    """How far the hips rise from the plant to the top, in torso lengths.

    We subtract in pixels first, then divide by the average torso length of
    the two frames. (Subtracting the torso-length values directly would mix
    two different scales if the player moved toward or away from the camera.)
    """
    rise_px = hip_pixels(top_row) - hip_pixels(plant_row)
    torso_px = (plant_row["torso_px"] + top_row["torso_px"]) / 2
    return rise_px / torso_px


def fmt(value, unit="", digits=1):
    """Format a number for printing, or say it was not tracked."""
    if value is None:
        return "not tracked"
    return f"{value:.{digits}f}{unit}"


def percent_tracked(rows, column):
    if not rows:
        return 0
    return 100 * sum(1 for r in rows if r[column] is not None) / len(rows)


def main():
    if len(sys.argv) < 2:
        print("Usage: python find_contact.py <name>_angles.csv")
        sys.exit(1)

    rows = load_rows(sys.argv[1])

    jump = find_jump_peak(rows)
    if jump is None:
        print("The hips were never tracked, so the jump cannot be found.")
        sys.exit(1)
    print(f"Jump detected around frame {jump['frame']} (time {jump['time_sec']:.2f} s)")
    print()

    contact, highest = find_contact(rows, jump)
    if contact is None:
        print("The hitting wrist was not tracked near the jump, so contact cannot be found.")
        sys.exit(1)

    c = contact["frame"]

    print(f"Contact frame: {c} (time {contact['time_sec']:.2f} s)")
    if contact is not highest:
        print(
            f"  Note: frame {highest['frame']} had the highest wrist, but frame {c} "
            f"was almost as high with a straighter elbow, so it was picked (tie-break)."
        )
    print(f"  Elbow angle:            {fmt(contact['elbow_angle'], ' deg')}  (180 = straight arm)")
    print(f"  Wrist above head:       {fmt(contact['wrist_above_head'], ' torso lengths', 2)}")

    ahead = contact["wrist_ahead_of_shoulder"]
    if ahead is None:
        position = "not tracked"
    elif ahead > 0:
        position = f"IN FRONT of the shoulder by {ahead:.2f} torso lengths"
    else:
        position = f"BEHIND the shoulder by {-ahead:.2f} torso lengths"
    print(f"  Wrist position:         {position}")

    lean = contact["trunk_lean"]
    lean_word = "" if lean is None else (" (forward)" if lean > 0 else " (back)")
    print(f"  Trunk lean:             {fmt(lean, ' deg')}{lean_word}")

    # At 30 fps the arm moves a lot between frames, so the real contact moment
    # may be between two frames. Show the frames around it so we can see that.
    print()
    print("Frames around contact (check how much the numbers change):")
    print("  frame  wrist_height  elbow  above_head  ahead  lean")
    for r in rows:
        if abs(r["frame"] - c) <= NEIGHBOR_FRAMES:
            mark = "  <- contact" if r["frame"] == c else ""
            print(
                f"  {r['frame']:>5}  {fmt(r['wrist_height'], '', 2):>12}"
                f"  {fmt(r['elbow_angle'], '', 0):>5}"
                f"  {fmt(r['wrist_above_head'], '', 2):>10}"
                f"  {fmt(r['wrist_ahead_of_shoulder'], '', 2):>5}"
                f"  {fmt(r['trunk_lean'], '', 0):>4}{mark}"
            )

    # Always report how well the arm was tracked near contact
    nearby = [r for r in rows if abs(r["frame"] - c) <= TRACKING_WINDOW]
    print()
    print(
        f"Tracking in frames {c - TRACKING_WINDOW} to {c + TRACKING_WINDOW}: "
        f"wrist {percent_tracked(nearby, 'wrist_height'):.0f}%, "
        f"elbow {percent_tracked(nearby, 'elbow_angle'):.0f}%"
    )

    # ----- Jump: plant, knee bend, and jump height -----
    print()
    plant, lowest_hips = find_plant(rows, jump)
    top = find_top_of_jump(rows, plant) if plant else None
    if plant is None or top is None:
        print("The hips were not tracked well enough to measure the jump.")
        return

    p = plant["frame"]
    print(f"Plant (hips lowest before takeoff): frame {p} (time {plant['time_sec']:.2f} s)")
    if plant is not lowest_hips:
        print(
            f"  Note: frame {lowest_hips['frame']} had the lowest hips, but frame {p} "
            f"was almost as low with a more bent knee, so it was picked (tie-break)."
        )
    print(f"  Knee angle at plant:    {fmt(plant['knee_angle'], ' deg')}  (180 = straight leg)")
    print(f"Top of jump (hips highest): frame {top['frame']} (time {top['time_sec']:.2f} s)")
    print(f"  Hip rise, plant to top: {jump_height(plant, top):.2f} torso lengths")
    print("  (This includes the knee bend at the plant, so it is bigger than")
    print("   the jump measured from normal standing height.)")

    # Hip height changes very little around the plant, but the knee angle can
    # change a lot, so show the frames around it too.
    print()
    print("Frames around the plant:")
    print("  frame  hip_px  knee")
    for r in rows:
        if abs(r["frame"] - p) <= NEIGHBOR_FRAMES and r["hip_height"] is not None:
            mark = "  <- plant" if r["frame"] == p else ""
            print(f"  {r['frame']:>5}  {hip_pixels(r):>6.1f}  {fmt(r['knee_angle'], '', 0):>4}{mark}")

    nearby = [r for r in rows if abs(r["frame"] - p) <= TRACKING_WINDOW]
    print()
    print(
        f"Tracking in frames {p - TRACKING_WINDOW} to {p + TRACKING_WINDOW}: "
        f"hip {percent_tracked(nearby, 'hip_height'):.0f}%, "
        f"knee {percent_tracked(nearby, 'knee_angle'):.0f}%"
    )


if __name__ == "__main__":
    main()
