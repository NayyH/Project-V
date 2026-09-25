# Volleyball Coach App

## Goal
An app that analyzes a player's volleyball movement from video and gives
feedback to help them improve. Users have accounts, and results are saved
in a database so they can track progress over time.

## Planned stack
- App: Expo (development build), React Native, TypeScript
- Accounts, database, video storage: Supabase (Postgres, Row Level Security on)
- Pose analysis: Python + MediaPipe Pose, later served with FastAPI

## Planned database tables
profiles, skills, sessions, metrics, feedback, targets
(targets stores the "good form" ranges so advice can change without app updates)

## Current phase: Phase 1 (prove the AI works)
No app yet. Only Python scripts. First skill: the spike (attack hit).
Side camera view, one hitter, hitting arm is a script argument (left or right).

## Spike phases to detect
1. Approach (steps toward the net)
2. Plant and load (last two steps, knees bend, arms swing back)
3. Jump (hips rise)
4. Arm cock (hitting elbow high and back, wrist behind the head)
5. Contact (hitting wrist at its highest point, arm nearly straight)
6. Landing

## Numbers to measure
- Elbow angle at contact (good is close to straight)
- Contact height: how high the wrist is above the head or standing reach
- Contact position: is the wrist in front of the shoulder or behind it
- Knee bend at the plant (before the jump)
- Jump height: how far the hips rise, compared to torso length
- Trunk lean at contact

## Important notes
- Camera distance changes pixel sizes, so compare things to the player's
  torso length instead of raw pixels.
- Hitters move fast and MediaPipe may lose the arm. Always report the percent
  of frames tracked and ignore low-confidence points.
- The ball is not tracked in Phase 1. Contact is estimated from the arm.

## Rules
- Keep code simple and commented. I am a junior developer and want to learn.
- Explain what you changed in short, plain sentences.
- Do not add libraries without telling me why.
- Do not use em dashes in any text or comments.