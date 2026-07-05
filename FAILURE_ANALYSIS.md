# Recovery-Start Failure Analysis

This analysis summarizes recovery-start CSV files under
`logs/narrow_passage_eval/`. It focuses on scenarios that initialize the robot
from difficult in-corridor states:

- `left_wall` / `left_wall_start`
- `right_wall` / `right_wall_start`
- `yaw_left` / `yaw_left_start`
- `yaw_right` / `yaw_right_start`

The numbers below aggregate all matching historical CSV rows currently present
in the log directory. They should be interpreted as a failure-mode diagnostic.
For a final paper table, the same analysis should be repeated on a fixed
checkpoint set with matched trial counts.

## Summary

| Scenario | Trials | Success rate | Failure rate | Collision rate | Wedge rate | Timeout rate | Rejected rate | Main failure type |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| left_wall | 1216 | 0.299 | 0.701 | 0.572 | 0.000 | 0.021 | 0.000 | collision |
| right_wall | 384 | 0.198 | 0.802 | 0.677 | 0.010 | 0.047 | 0.000 | collision |
| yaw_left | 1024 | 0.191 | 0.809 | 0.512 | 0.001 | 0.000 | 0.000 | collision |
| yaw_right | 320 | 0.144 | 0.856 | 0.434 | 0.000 | 0.250 | 0.000 | collision + timeout |

The highest-failure scenario is `yaw_right`, with an aggregated failure rate of
0.856. The next most difficult scenarios are `yaw_left` and `right_wall`, both
above 0.80 failure rate in the current recovery-start logs.

## Failure Types

Collision is the dominant failure mode across the recovery-start set. This is
especially visible for `right_wall`, where the robot starts close to the wall
and often fails before it can create enough lateral clearance. `left_wall` also
shows collision-dominated failures, but its success rate is higher than the
right-wall and yaw-start cases.

Yaw perturbations are also difficult. `yaw_left` has a high collision rate,
suggesting that heading error quickly turns into wall contact inside the narrow
passage. `yaw_right` has the highest overall failure rate; its failures are
split between collision and timeout, which indicates that the policy often
either contacts the wall or becomes too conservative/slow to finish cleanly.

Wedge and rejected events are rare in the current CSV set. This does not mean
they are unimportant; rather, the observed recovery-start failures are mostly
manifesting as contact and timeout before a clear wedge or reject label is
triggered.

## Interpretation

The nominal narrow-passage evaluations show that the low-level PPO locomotion
policy is effective when the robot starts from a clean passage entrance. In that
setting, the controller can track forward progress, maintain reasonable
clearance, and traverse narrow straight corridors with high success.

The recovery-start logs reveal a different regime. When the robot is already
yawed, near a wall, or initialized from a partially failed state inside the
corridor, the same memory-free low-level controller is less stable. The failure
rates are much higher, and the dominant failure type is collision with the
passage boundary.

This should be treated as a research motivation rather than a project defect.
The result clarifies the boundary of low-level locomotion RL: PPO can learn a
useful narrow-passage gait, but severe recovery states require additional
structure around the gait controller. A complete narrow-passage autonomy stack
should combine this low-level controller with higher-level risk judgment,
failure-state memory, and recovery strategy selection. Those higher-level
components can decide when to slow down, reject entry, back out, re-align,
switch recovery mode, or avoid repeated attempts after a failed passage.

In this framing, the recovery-start failures are valuable evidence. They show
where low-level control is sufficient, where it breaks down, and why future work
should add a failure-aware layer above the locomotion policy.
