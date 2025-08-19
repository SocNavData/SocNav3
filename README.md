# SocNav3 dataset code and tools

Official implementation of the paper [Towards Data-Driven Metrics for Social Robot Navigation Benchmarking](...)

This project is a joined effort towards the development of a data-driven Social Robot Navigation metric to facilitate benchmarking and policy optimization. Motivated by the lack of a standardized method to benchmark Social robot Navigation (SocNav) policies, we propose an **A**ll-encompassing **L**earned **T**rajectory-wise (ALT) metric, which is learned directly from human evaluations of full robot trajectories, conditioned on contexts.

## Dataset

### Data stored per trajectory

| **Type**       | **Variable**            | **Description** |
|----------------|-------------------------|-----------------|
| **Robot**      | Pose                    | 2D position (m) and orientation (rad) on the plane. |
|                | Speed                   | Lineal (m/s), angular (rad/s). |
|                | Drive                   | Categorical (differential / omni / ackerman). |
|                | Shape                   | 2D *circle* (radius), *rectangle* (width, height), or *polygon* (list of points). |
| **Task**       | Type                    | Task type, either *“go-to”*, *“guide-to”* or *“follow”*. |
|                | Position + threshold    | For go-to and guide-to tasks. 2D position + threshold (m). |
|                | Orientation + threshold | For go-to and guide-to tasks. Orientation + threshold (rad). |
|                | Human identifier        | For *guide-to* and *follow* tasks. |
|                | Context                 | Textual description of the context, in English. |
| **Humans**     | Identifier              | Integer that uniquely identifies the human in a given episode. |
|                | Pose                    | 2D position (m) and orientation (rad) on the plane. |
|                | Full Pose (optional)    | The 3D position of the COCO-18 key point set. |
| **Objects**    | Identifier              | Integer that uniquely identifies the object in a given episode. |
|                | Type                    | Free text describing the type of the object. |
|                | Pose                    | 2D position (m) and orientation (rad) on the plane. |
|                | Shape                   | The 2D shape of the object. |
| **Environment**| Walls                   | Sequence of polylines (2D, m). |
|                | Grid                    | Occupancy map: 2D grid + resolution (m/cell). |
|                | Area semantics          | Free text describing the area, e.g., “indoor”, “outdoor”, “kitchen”, “a science museum”. |

## Tools

## Baseline

## How to contribute