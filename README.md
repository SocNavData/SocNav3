# SocNav3 dataset code and tools

Official implementation of the paper [Towards Data-Driven Metrics for Social Robot Navigation Benchmarking](...)

This project is a joined effort towards the development of a data-driven Social Robot Navigation metric to facilitate benchmarking and policy optimization. Motivated by the lack of a standardized method to benchmark Social robot Navigation (SocNav) policies, we propose an **A**ll-encompassing **L**earned **T**rajectory-wise (ALT) metric, which is learned directly from human evaluations of full robot trajectories, conditioned on contexts.

This repository contains the core code and utilities for working with the proposed trajectory-wise dataset (SocNav3). Alongside a baseline implementation to replicate our results, it also includes tools to check and visualize data, supporting dataset extension and further research.

## Dataset

The dataset comprises variables related to raters, trajectories, and rater-trajectory scores. For every rater, together with demographic information, a rating list is stored. The rating list contains tuples ($t$, $c$, $r$), where $t$ is a trajectory identiﬁer (string), $c$ is a context (string), and a $r$ is a score assigned by the rater to the trajectory $t$ given the context $c$.

Trajectories contain data on the robot, the task and its context, humans, objects, and the environment. Except for variables related to the task and the environment, which apply to the whole trajectory, variables are recorded with a timestamp at each time step. Next section includes a description of the data stored per trajectory.

Contexts are textual descriptions of the specific situations that occur during the trajectory. They are not necessarily bound to trajectories when trajectories are recorded; this enables us to vary the context to explore how different contexts affect the perception of the same trajectory.

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

### Data organization

The raw dataset contains two main directories: one containing the trajectories and another one containing data about the raters and their ratings. 

The trajectories directory contains JSON ﬁles of recorded trajectories in sub-directories named according to the source of the trajectory data. Each ﬁle contains one trajectory with the structure described in the previous section. There is an additional file ($trajectory_variants.json$) that groups together all the variants of each trajectory (e.g., corresponding to the same scenario with different walls' configurations.)

The ratings directory contains a separate JSON ﬁle for each rater. The rating list includes control questions that allow analyzing the consistency of the data.

The raw trajectories' dataset can be found at [SocNav3_all_data/dataset/unlabeled](https://www.dropbox.com/scl/fo/ze7op896sqb5tog89xnpl/AEm4g0fbyV_71tpR1ESH7Ic?rlkey=fqngtvz58uf26fbvwkjgnulgi&st=bcn5zuwl&dl=0). The ratings are available at [SocNav3_all_data/ratings](https://www.dropbox.com/scl/fo/yybho991ousbt1grhnd1s/ABwXfRZpGrIQ_Tx8RK9XUBM?rlkey=98qgq181jbdeyaix4ofc4fk4v&st=jw496ojt&dl=0). This last folder contains all the ratings and a selected set of ratings resulting from a consistency analysis.

After downloading trajectories and ratings, a labeled dataset can be obtained by running the following commands:

```shell
cd dataset
python3 label_dataset.py --trajectories PATH_TO_THE_TRAJECTORIES_DIRECTORY --ratings PATH_TO_THE_RATINGS_DIRECTORY
```
A labeled version of the dataset can be directly downloaded from [SocNav3_all_data/dataset/unlabeled](https://www.dropbox.com/scl/fo/go4ud504exi7yr7sq1mwy/AKSA84sasbsPIvOjP78zdoY?rlkey=t0cwl3g6p8cfk8akxnhfnuzeq&st=j89cykq5&dl=0)

Once the labeled dataset is generated, it can be used for training a model producing an ALT-metric. For that, the whole dataset can be split into train/validation/test sets using the script dataset/split_dataset.py as follows:

```shell
cd dataset
python3 split_dataset.py --dataset PATH_TO_THE_LABELED_DATASET
```
The default split is 0.9/0.05/0.05. It can be modified using the arguments $trainpercentage$ and $valpercentage$.

## Tools

## Baseline

## How to contribute