from morphgym.structure.data import data



@data
class MorphologyConfig:
    max_limbs = 12
    max_joints_per_limb = 2


@data
class Mask:
    limb = None
    dense_joint = None
    joint = None

@data
class MorphologyInfo:
    xml_path: list = ()
    geom_dim: int = 6
    joint_dim: int = 9