from morphgym.utils import ODict


from collections import Iterable
from morphgym.structure.space.morphology_space import MorphologySpace
from lxml import etree
from gymnasium.spaces.utils import OrderedDict,flatten
from gymnasium.spaces import Tuple
from morphgym.utils import print_dict
import math

from morphgym.utils import ODict

from morphgym.morphology.data import MorphologyConfig,Mask,MorphologyInfo

geom_type_map = {
    "sphere": 0,
    "capsule": 1
}

axis_idx_map = {
    'x': 0,
    'y': 1,
    'z': 2
}

idx_axis_map = {
    0:'x',
    1:'y',
    2:'z'
}


def empty_joint():
    return OrderedDict({
            'axis': (0., 0., 0.),
            'pos': (0., 0., 0.),
            'range': (0., 0.),
            'gear': 0.,
        })


def empty_limb():
    return OrderedDict({
        "parent_idx": -2,
        "idx": -2,
        "geom": OrderedDict({
            "type": 0.,
            "size": (0., 0.),
            "mass": 0.,
        }),
        'joints': (empty_joint(),empty_joint())
    })




class Morphology(list):
    init_mode = ('dict','xml_path')

    def __init__(self, morphology_cfg=None, dict=None, xml_path: list = None):
        """
        Vector Morphology.

        morphology.
        Agent observation space, action space.
        """

        # set data
        self.cfg = morphology_cfg if morphology_cfg is not None else MorphologyConfig()
        self.mask = Mask()
        self.info = MorphologyInfo()

        mode_count = 0
        for mode in Morphology.init_mode:
            mode_count += (locals()[mode] is not None)
        if mode_count >= 2:
            raise ValueError(f'please choose only one init mode from: {Morphology.init_mode}')

        if dict is not None:
            self.extend(dict)
            self.xml_list = [morphology.xml_list for morphology in dict]
        if xml_path is not None:
            self.xml_list = xml_path
            self.load_xml(xml_path)

        # calculate limbs
        self.max_limbs = morphology_cfg.max_limbs


        self.morphology_space =  Tuple([MorphologySpace(morphology_cfg=self.cfg)
                                        for _ in range(len(self))])



    def load_xml(self, xml_list):
        if isinstance(xml_list, str):
            xml_list = [xml_list]
        else:
            try:
                all(isinstance(y, str) for y in xml_list)
            except:
                raise TypeError("xml_list must be a list of xml path (list) or a single xml path (string).")

        self.clear()
        self.mask.limb = []
        self.mask.joint = []
        self.mask.dense_joint = []

        for xml in xml_list:

            parser = etree.XMLParser(remove_blank_text=True)
            tree = etree.parse(xml, parser)
            root = tree.getroot()
            worldbody = root.find('worldbody')
            firstbody = worldbody.find('body')
            actuator = root.find('actuator')
            joint_gear_map = {}
            for motor in root.find('actuator').getchildren():
                joint_gear_map[motor.attrib['joint']] = motor.attrib['gear']

            limbs = []

            # tree.write(xml, xml_declaration=True, encoding="utf-8", pretty_print=True)
            def get_limb(body, parent_idx, limb_list, limb_mask, joint_mask):
                children_body_list = []
                idx = len(limb_list)

                geom = body.find("geom")
                geom_type = geom_type_map[geom.attrib["type"]]
                geom_size = float(geom.attrib["size"])
                if geom.attrib["type"] == "sphere":
                    geom_size2 = 0.
                    geom_volume = 4 / 3 * math.pi * math.pow(geom_size, 3)
                else:
                    geom_size2 = abs(float(geom.attrib["fromto"].split()[-1]) / 2)
                    geom_volume = 4 / 3 * math.pi * math.pow(geom_size, 3) + 8 * geom_size2 * math.pow(geom_size2, 2)
                geom_mass = geom_volume

                limb_dict = OrderedDict({
                    "parent_idx": parent_idx,
                    "idx": idx,
                    "geom": OrderedDict({
                        "type": geom_type_map[body.find("geom").attrib["type"]],
                        "size": (geom_size, geom_size2),
                        "mass": geom_mass,
                    })
                })

                # get joints info
                joints = []

                for child in body.getchildren():
                    if child.tag == 'body':
                        children_body_list.append(child)
                    if child.tag == 'joint':
                        joints.append(OrderedDict({
                            'axis': list(map(float, child.attrib['axis'].split())),
                            'pos': list(map(float, child.attrib['pos'].split())),
                            'range': list(map(float, child.attrib['range'].split())),
                            'gear': joint_gear_map[child.attrib['name']],
                        }))
                        joint_mask.append(True)

                while len(joints) < self.cfg.max_joints_per_limb:
                    joints.append(empty_joint())
                    joint_mask.append(False)

                limb_dict['joints'] = joints
                limb_list.append(limb_dict)
                limb_mask.append(True)
                for child_body in children_body_list:
                    get_limb(child_body, idx, limb_list, limb_mask, joint_mask)

            limb_mask = []
            dense_joint_mask = []
            get_limb(firstbody, -1, limbs, limb_mask, dense_joint_mask)

            joint_mask = dense_joint_mask[:]
            # flatten(MorphologySpace(len(limb_list)),limb_list)

            while len(limbs) < self.cfg.max_limbs:
                limbs.append(empty_limb())
                limb_mask.append(False)
                joint_mask.extend([False for _ in range(self.cfg.max_joints_per_limb)])

            self.mask.limb.append(limb_mask)
            self.mask.joint.append(joint_mask)
            self.mask.dense_joint.extend(dense_joint_mask)

            self.append(limbs)

        self.info.xml_list = xml_list

    @property
    def flat(self):
        return flatten(self.morphology_space ,self)
