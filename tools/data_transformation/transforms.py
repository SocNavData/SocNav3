from data_mirroring import mirror_sequence
from data_normalization import transform_to_goal_fr
from data_random_orientation import transform_with_random_orientation
from data_random_noise import transform_with_random_noise


class NormalizeTrajectory:
    def __call__(self, trajectory):
        return transform_to_goal_fr(trajectory)

class ApplyMirroring:
    def __call__(self, trajectory):
        return mirror_sequence(trajectory)

class ApplyRandomOrientation:
    def __call__(self, trajectory):
        return transform_with_random_orientation(trajectory)

class ApplyRandomNoise:
    def __call__(self, trajectory):
        return transform_with_random_noise(trajectory)
