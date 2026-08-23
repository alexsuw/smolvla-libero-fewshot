# Failure cases

The paper uses three real failed zero-shot rollouts from the frozen seen policy,
all at evaluation seed 1000. Four frames from each video are stored under
`report/latex/imgs/failure_*.jpg`.

## 1. Drawer: spatial grounding

The arm moves over the work surface but never reaches the middle drawer handle.
Hypothesis: the instruction is not grounded to the correct handle. Separating
test: compare original, handle-cropped, and point-marked images while measuring
pre-contact end-effector distance.

## 2. Bowl: grasping or relational planning

The arm reaches a plausible central area but does not grasp and place the bowl.
Hypothesis: grasping is the first bottleneck. Separating test: compare the normal
start with a matched start where the bowl is already in the gripper.

## 3. Wine: object identity or placement horizon

The first reach is directed toward a distractor region and the bottle is not
picked. Separating test: cross a colour swap with a bottle-already-grasped start.
This separates object identity from the cabinet-top placement stage.

## Language control

Correct instructions give 1/60 success and wrong instructions 0/60 on identical
initial-state fingerprints. Mean paired action L2 distances are 0.582 (Drawer),
0.998 (Bowl), and 0.966 (Wine): language changes trajectories even though
zero-shot competence is weak.

See the [main paper](latex/build/paper.pdf), Section 4, for the frames and full
experimental designs.
