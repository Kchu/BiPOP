#!/usr/bin/env python
"""
Tasks of NICOL in CoppeliaSim: ServeWater, ServeFruit and StackBlock (block layout "mix"; variants StackBlock-simple, -complex, -super).

Each task loads its model into the scene, resets the scene from a task config
(or samples a new random config) and checks the task goal with the proximity
sensors of the scene.

Importing this module connects to CoppeliaSim (NICOL/nicol.ttt must be loaded).
"""
import itertools
import os
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .nicol_sim import NicolPose, create_nicol

# =====================================================================
# INITIALIZATION
# =====================================================================
PATH = os.path.dirname(os.path.abspath(__file__))

nicol = create_nicol()
head = nicol.head()
left = nicol.left()
right = nicol.right()
sim = nicol.sim

random.seed(666)  # reproducible random task configs


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

# Camera configs keyed by task family. Each position/orientation is chosen to frame
# the full workspace: robot arms, table objects, and the human hand-off zone.
_CAMERA_CONFIGS: Dict[str, Dict] = {
    # Front-elevated isometric view: captures both arms across the full table width
    # and the human positioned at ~1.1 m depth, looking slightly down the X-axis.
    "ServeWater": {
        "position":    [3.110799492530379, -0.05373128594745438, 3.19741080564999],
        "orientation": [3.141592653583946, -0.8172226446907499, 1.5707963267868794],
    },
    # Right-side diagonal view: the fruit/bowl task involves wide lateral reach,
    # so an off-axis angle better shows bimanual Y-separation and bowl placement.
    "ServeFruit": {
        "position":    [2.553935327267101, -2.2898812546896194, 2.326125761887166],
        "orientation": [-2.155685319987217, -0.5987280998163329, 2.784402136938067],
    },
    # Same front-elevated view as ServeWater; blocks are clustered on the table
    # and the tray/serve points are within the same XY footprint.
    "StackBlock": {
        "position":    [2.8133590693780897, -0.9588375585603853, 2.786163798493663],
        "orientation": [-2.6654727922810943, -0.8051362527501763, 2.191727445742949],
    },
}


def setup_camera(task_name: str) -> None:
    """Set the default camera to the best angle for the given task."""
    family = task_name.split("-")[0]  # e.g. "StackBlock-simple" -> "StackBlock"
    cfg = _CAMERA_CONFIGS[family]
    cam = sim.getObject('/DefaultCamera')
    sim.setObjectPosition(cam, -1, cfg["position"])
    sim.setObjectOrientation(cam, -1, cfg["orientation"])


def get_object_safe(name: str) -> Optional[int]:
    """Get object handle safely, return None if not found."""
    try:
        return sim.getObject(name)
    except Exception:
        return None


def remove_model_safe(name: str) -> None:
    """Remove model if it exists."""
    try:
        sim.removeModel(sim.getObject(name))
    except Exception:
        pass


def set_static(handle: int, static: bool = True, respondable: bool = True) -> None:
    """Set object static and respondable parameters."""
    sim.setObjectInt32Parameter(handle, sim.shapeintparam_static, 1 if static else 0)
    if not respondable:
        sim.setObjectInt32Parameter(handle, sim.shapeintparam_respondable, 0)


def determine_area(y: float, threshold: float = 0.0) -> str:
    """Determine left/right/overlap area based on y coordinate."""
    if y > threshold:
        return "left"
    elif y < -threshold:
        return "right"
    return "overlap"


# =====================================================================
# BASE TASK CLASS
# =====================================================================
class BaseTask(ABC):
    """Abstract base class for all tasks."""
    
    def __init__(self, name: str, other_models: List[str]):
        # Remove other task models
        for model in other_models:
            remove_model_safe(f'/{model}')
        
        self.name = name
        self.sim = sim
        setup_camera(name)
        self.positions: Dict[str, Any] = {}
        self.areas: Dict[str, str] = {}
        self.config: Dict[str, Any] = {}
        self.type: Any = None
        self.task_des: str = ""
        self.task_des_visual: str = ""
        
        # Common objects
        self.human = get_object_safe('/Bill')
        self.human_goal = get_object_safe('/goalDummy')
    
    @abstractmethod
    def _load_objects(self) -> None:
        """Load task-specific objects."""
        pass
    
    @abstractmethod
    def reset(self, config: Optional[Dict] = None) -> None:
        """Reset task to initial state."""
        pass
    
    @abstractmethod
    def check_success(self) -> bool:
        """Check if task is successfully completed."""
        pass
    
    def get_image_from_eyes(self, index: int):
        """Capture image from robot's eyes."""
        head.set_pose_target(NicolPose([0.75, 0.0, 1.1], [0, 0, 0, 0]))
        image = nicol.head_.get_left_eye_camera_img(False, prefix=f"{self.name}_{index}")
        head.set_pose_target(NicolPose([0.75, 0.0, 1.4], [0, 0, 0, 0]))
        return image
    
    def _set_human_pose(self, y: float) -> None:
        """Set human position and determine area."""
        sim.setObjectPose(self.human, sim.handle_world, [3, y, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [3, y, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y, 0, 0, 0, -1, 0])
        self.positions["human"] = [1.1, y, 0]
        self.areas["human"] = determine_area(y, threshold=0.1)


# =====================================================================
# SERVE WATER TASK
# =====================================================================
class ServeWaterTask(BaseTask):
    """Task: Pour water from blue cup to yellow cup, serve yellow cup."""
    
    def __init__(self):
        super().__init__("ServeWater", ["ServeFruit", "StackBlock"])
        self.text_queries = ["human face", "blue cup", "yellow cup", "brown box", "red ball"]
        self.owl_threshold = 0.04
        self._load_objects()
    
    def _load_objects(self) -> None:
        """Load ServeWater task objects."""
        if not get_object_safe("/Origin_blue_cup"):
            sim.loadModel(f"{PATH}/ServeWater.ttm")
        
        self.blue_cup = sim.getObject("/Origin_blue_cup")
        self.yellow_cup = sim.getObject("/Origin_yellow_cup")
        self.ball = sim.getObject("/big_ball")
        self.serve_point = sim.getObject("/serve_point")
        self.store_point = sim.getObject("/store_point")
        self.box = sim.getObject("/bbox")
        self.sensor_ball = sim.getObject("/yellow_cup_sensor")
        self.sensor_serve = sim.getObject("/serve_point_sensor")
        self.sensor_box = sim.getObject("/bbox_sensor")
        
        # Set static properties
        for cup in ["/blue_cup_respondable", "/yellow_cup_respondable"]:
            set_static(sim.getObject(cup))
        set_static(self.ball, respondable=False)
    
    def reset(self, config: Optional[Dict] = None) -> None:
        """Reset task with optional config."""
        if config:
            self._reset_from_config(config)
        else:
            self._reset_random()

    @staticmethod
    def _normalize_serve_y(y: float) -> float:
        """Normalize serve_point y-coordinate for side positions."""
        if y == -0.5:
            return -0.3
        if y == 0.5:
            return 0.3
        return y
    
    def _reset_from_config(self, config: Dict) -> None:
        """Reset using provided configuration."""
        self.config = config
        self.positions = config["objects"]
        self.areas = config["area"]
        self.type = config["type"]
        
        y_human = self.positions["human"][1]
        self._set_human_pose(y_human)
        
        serve_pos = list(self.positions["serve_point"])
        serve_pos[1] = self._normalize_serve_y(serve_pos[1])
        sim.setObjectPose(self.serve_point, sim.handle_world, 
                 serve_pos + [0, 0, 0, 1])
        sim.setObjectPose(self.sensor_serve, sim.handle_world,
                 serve_pos + [0, 0, 0, 1])
        self.positions["serve_point"] = serve_pos
        self.positions["serve_point_sensor"] = list(serve_pos)
        time.sleep(2)
        
        sim.setObjectPose(self.box, sim.handle_world, 
                         self.positions["box"] + [0, -0.7, 0, 0.7])
        sim.setObjectPose(self.store_point, self.box, 
                         self.positions["store_point"] + [0, 0, 0, 1])
        sim.setObjectPose(self.blue_cup, sim.handle_world, 
                         self.positions["blue_cup"] + [0, 0, 0, 1])
        sim.setObjectPose(self.yellow_cup, sim.handle_world, 
                         self.positions["yellow_cup"] + [0, 0, 0, 1])
        
        # Ball in blue cup
        parent = sim.getObjectHandle('/blue_cup_respondable')
        sim.setObjectParent(self.ball, parent, True)
        sim.setObjectPosition(self.ball, parent, [0.03, -0.01, -0.02])
    
    def _reset_random(self) -> None:
        """Reset with random configuration."""
        # Human position
        y_human = random.choice([0.5, -0.5, 0])
        self._set_human_pose(y_human)
        self.areas["human"] = determine_area(y_human, threshold=0.1)

        serve_y = self._normalize_serve_y(y_human)
        serve_pos = [0.85, serve_y, 1.1]
        
        sim.setObjectPose(self.serve_point, sim.handle_world, 
                 serve_pos + [0, 0, 0, 1])
        sim.setObjectPose(self.sensor_serve, sim.handle_world,
                 serve_pos + [0, 0, 0, 1])
        self.positions["serve_point"] = serve_pos
        self.positions["serve_point_sensor"] = list(serve_pos)
        time.sleep(2)
        
        # Cup positions
        z = 0.86
        x1, y1 = random.choice([[0.4, 0.4], [0.4, -0.4], [0.4, 0]])
        x2, y2 = random.choice([[0.7, 0.4], [0.7, -0.4], [0.7, 0]])
        
        self.areas["blue cup"] = determine_area(y1)
        self.areas["yellow cup"] = determine_area(y2)
        
        sim.setObjectPose(self.blue_cup, sim.handle_world, [x1, y1, z, 0, 0, 0, 1])
        sim.setObjectPose(self.yellow_cup, sim.handle_world, [x2, y2, z, 0, 0, 0, 1])
        self.positions["blue_cup"] = [x1, y1, z]
        self.positions["yellow_cup"] = [x2, y2, z]
        
        # Box position
        box_pose = random.choice([[0.55, 0.6, 0.83], [0.55, -0.6, 0.83]])
        self.positions["box"] = box_pose
        self.areas["brown box"] = "left" if box_pose[1] > 0 else "right"
        
        sim.setObjectPose(self.box, sim.handle_world, box_pose + [0, -0.7, 0, 0.7])
        
        store_offset = [0.06, 0.025 if box_pose[1] > 0 else -0.025, 0]
        sim.setObjectPose(self.store_point, self.box, store_offset + [0, 0, 0, 1])
        self.positions["store_point"] = store_offset
        
        # Ball in blue cup
        parent = sim.getObjectHandle('/blue_cup_respondable')
        sim.setObjectParent(self.ball, parent, True)
        sim.setObjectPosition(self.ball, parent, [0.03, -0.01, -0.02])
        self.positions["ball"] = 1
        self.areas["red ball"] = "blue cup"
        
        self.type = [
            f"blue: {self.areas['blue cup']}",
            f"yellow: {self.areas['yellow cup']}",
            f"human: {self.areas['human']}",
            f"store: {self.areas['brown box']}",
            f"ball: {self.areas['red ball']}"
        ]
        
        self._generate_descriptions()
        self.config = {
            "objects": self.positions,
            "area": self.areas,
            "task_des": self.task_des,
            "task_des_visual": self.task_des_visual,
            "type": self.type
        }
    
    def _generate_descriptions(self) -> None:
        """Generate task descriptions."""
        self.task_des = """
### Task Information: 
## Task Init: There are two cups on the table: the blue_cup contains water and the yellow_cup is empty.  
## Task goal: Move the yellow_cup with water to the serve_point (in front of human), and release blue_cup on table at the store_point (on brown box). It is not allowed to release objects at the serve_point. 

## Task related objects: blue_cup, yellow_cup 
## Task related points: store_point, serve_point, overlap_point_1, overlap_point_2
"""
        self.task_des_visual = f"""
# Based on the vision analyzation, the human is standing in front of the {self.areas['human']} area, the blue cup is at the {self.areas['blue cup']} area, the yellow cup is at the {self.areas['yellow cup']} area, the brown box is at the {self.areas['brown box']} area.
"""
    
    def check_success(self) -> bool:
        """Check if water is in yellow cup at serve point, blue cup in box."""
        yellow_respond = sim.getObject("/yellow_cup_respondable")
        blue_respond = sim.getObject("/blue_cup_respondable")
        
        # Get hand sensors
        left_sensor = sim.getObject('/l_sensor')
        right_sensor = sim.getObject('/r_sensor')
        
        ball_in_yellow = sim.checkProximitySensor(self.sensor_ball, self.ball)[0]
        yellow_at_serve = sim.checkProximitySensor(self.sensor_serve, yellow_respond)[0]
        blue_in_box = sim.checkProximitySensor(self.sensor_box, blue_respond)[0]
        
        # Check if either hand sensor detects the yellow cup
        left_hand_detects = sim.checkProximitySensor(left_sensor, yellow_respond)[0]
        right_hand_detects = sim.checkProximitySensor(right_sensor, yellow_respond)[0]
        hand_detects_yellow = left_hand_detects or right_hand_detects
        
        # Check if either hand sensor detects the blue cup (should NOT be grasped)
        left_hand_detects_blue = sim.checkProximitySensor(left_sensor, blue_respond)[0]
        right_hand_detects_blue = sim.checkProximitySensor(right_sensor, blue_respond)[0]
        hand_not_grasping_blue = not (left_hand_detects_blue or right_hand_detects_blue)
        
        # Yellow cup at serve point is successful if either serve sensor detects it 
        # or one of the hand sensors detects it while at serve point
        yellow_at_serve_success = yellow_at_serve and hand_detects_yellow
        
        return ball_in_yellow and yellow_at_serve_success and blue_in_box and hand_not_grasping_blue


# =====================================================================
# SERVE FRUIT TASK
# =====================================================================
class ServeFruitTask(BaseTask):
    """Task: Place fruits in bowl and serve to human."""
    
    def __init__(self):
        super().__init__("ServeFruit", ["ServeWater", "StackBlock"])
        self.text_queries = ["human face", "red bowl", "apple", "banana"]
        self.owl_threshold = 0.03
        self._load_objects()
    
    def _load_objects(self) -> None:
        """Load ServeFruit task objects."""
        if not get_object_safe("/Origin_bowl"):
            sim.loadModel(f"{PATH}/ServeFruit.ttm")
        
        self.bowl = sim.getObject("/Origin_bowl")
        self.bowl_respond = sim.getObject("/bowl_respondable")
        self.apple = sim.getObject("/apple")
        self.banana = sim.getObject("/banana")
        self.target_sensor = sim.getObject("/serve_point_sensor")
        
        # Set static properties
        for obj in [self.bowl_respond, self.apple, self.banana]:
            set_static(obj)
    
    def reset(self, config: Optional[Dict] = None) -> None:
        """Reset task with optional config."""
        # Reset human to center first
        sim.setObjectPose(self.human_goal, sim.handle_world, [3, 0, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human, sim.handle_world, [3, 0, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, 0, 0, 0, 0, -1, 0])
        
        if config:
            self._reset_from_config(config)
        else:
            self._reset_random()
    
    def _reset_from_config(self, config: Dict) -> None:
        """Reset using provided configuration."""
        self.positions = config["objects"]
        self.areas = config["area"]
        self.type = config["type"]
        
        sim.setObjectPose(self.apple, sim.handle_world, 
                         self.positions["apple"] + [0, 0, 0, 1])
        sim.setObjectPose(self.banana, sim.handle_world, 
                         self.positions["banana"] + [0, 0, -4, 1])
        sim.setObjectPose(self.bowl, sim.handle_world, 
                         self.positions["bowl"] + [0, 0, 0, 1])
    
    def _reset_random(self) -> None:
        """Reset with random configuration."""
        z_apple, z_banana, z_bowl = 0.82, 0.81, 0.83
        
        x1, y1 = random.choice([[0.4, 0.5], [0.4, -0.5]])
        x2, y2 = random.choice([[0.5, 0.65], [0.5, -0.65]])
        x3, y3 = random.choice([[0.6, 0.4], [0.6, -0.4], [0.6, 0]])
        
        sim.setObjectPose(self.apple, sim.handle_world, [x1, y1, z_apple, 0, 0, 0, 1])
        sim.setObjectPose(self.banana, sim.handle_world, [x2, y2, z_banana, 0, 0, -4, 1])
        sim.setObjectPose(self.bowl, sim.handle_world, [x3, y3, z_bowl, 0, 0, 0, 1])
        
        self.positions = {
            "apple": [x1, y1, z_apple],
            "banana": [x2, y2, z_banana],
            "bowl": [x3, y3, z_bowl]
        }
        
        self.areas = {
            "apple": "left" if y1 > 0 else "right",
            "banana": "left" if y2 > 0 else "right",
            "bowl": determine_area(y3, threshold=0.1)
        }
        
        self.type = [self.areas["apple"], self.areas["banana"], self.areas["bowl"]]
        self._generate_descriptions()
        
        self.config = {
            "objects": self.positions,
            "area": self.areas,
            "task_des": self.task_des,
            "task_des_visual": self.task_des_visual,
            "type": self.type
        }
    
    def _generate_descriptions(self) -> None:
        """Generate task descriptions."""
        self.task_des = """
### Task Information:
## Task Description: There are three objects at the table: an apple, a banana, and a large bowl (not graspable with one single hand).
## Task Goal: Serve the bowl with fruits inside at the serve_point, which is right in front of the human.
## Task related objects: apple, banana, bowl
## Task related points: serve_point
You are not allowed to release objects at the serve_point.
"""
        self.task_des_visual = f"""
# Based on the vision analyzation, the human is standing in the central area, with the red bowl at the {self.areas['bowl']} area, the apple at the {self.areas['apple']} area, and the banana at the {self.areas['banana']} area.
"""
    
    def check_success(self) -> bool:
        """Check if fruits and bowl are at serve point."""
        apple_ok = sim.checkProximitySensor(self.target_sensor, self.apple)[0]
        banana_ok = sim.checkProximitySensor(self.target_sensor, self.banana)[0]
        bowl_ok = sim.checkProximitySensor(self.target_sensor, self.bowl_respond)[0]
        return apple_ok > 0 and banana_ok > 0 and bowl_ok > 0


# =====================================================================
# STACK BLOCKS TASK
# =====================================================================
@dataclass
class BlockInfo:
    """Information about a single block."""
    handle: int
    sensor: int
    position: Tuple[float, float, float]


class StackBlockTask(BaseTask):
    """Task: Stack colored blocks according to specified pattern."""
    
    COLORS = ["red", "yellow", "green", "blue", "orange", "black"]
    DESCRIPTION_ORDER = ["red", "green", "blue", "yellow", "orange", "black"]  # order in the scene description
    CONFIGS = {
        "simple": {"blocks": 4, "options": ((2, 2), (3, 1), (4,))},
        "complex": {"blocks": 5, "options": ((3, 2), (4, 1), (5,))},
        "super": {"blocks": 6, "options": ((3, 3), (4, 2), (5, 1), (6,))},
        "mix": {"blocks": 6, "options": ((3, 3), (4, 2), (5, 1), (6,))},
    }
    INIT_STACK_RELATIVE_POS = [0, 0, -0.05]
    
    def __init__(self, config: str = "simple"):
        self.config_mode = config if config in self.CONFIGS else "simple"
        super().__init__(f"StackBlock-{self.config_mode}", ["ServeWater", "ServeFruit"])
        
        self.text_queries = ["human face", "red block", "green block", "blue block", 
                            "yellow block", "orange block", "black block"]
        self.owl_threshold = 0.01
        
        cfg = self.CONFIGS[self.config_mode]
        self.block_nums = cfg["blocks"]
        self.stacking_options = cfg["options"]
        
        self.blocks: Dict[str, BlockInfo] = {}
        self.stacks: List[List[str]] = []
        self.pile_desc: List[Dict[str, Any]] = []
        self.initial_stacks: List[List[str]] = []
        
        self._load_objects()
    
    def _load_objects(self) -> None:
        """Load StackBlock task objects."""
        if not get_object_safe('/serve_point_1'):
            sim.loadModel(f"{PATH}/StackBlock.ttm")
        
        self.serve_1 = sim.getObject('/serve_point_1')
        self.serve_1_sensor = sim.getObject('/serve_point_1_sensor')
        self.serve_2 = sim.getObject('/serve_point_2')
        self.serve_2_sensor = sim.getObject('/serve_point_2_sensor')
        self.tray = sim.getObject('/tray')
        self.tray_ori = sim.getObjectOrientation(self.tray, -1)
    
    def reset(self, config: Optional[Dict] = None) -> None:
        """Reset task with optional config."""
        if config:
            self._reset_from_config(config)
        else:
            self._reset_random()
        
        self._generate_descriptions()
        self.config = {
            "objects": self.positions,
            "goal": self.stacks,
            "initial_stacks": self.initial_stacks,
            "pile_desc": self.pile_desc,
            "area": self.areas,
            "task_des": self.task_des,
            "task_des_visual": self.task_des_visual,
            "type": self.type
        }
    
    def _reset_from_config(self, config: Dict) -> None:
        """Reset using provided configuration."""
        self.positions = config["objects"]
        self.stacks = config["goal"]
        self.areas = config["area"]
        self.type = config["type"]
        self.initial_stacks = config.get("initial_stacks", [])
        
        # Set block positions
        for color in self.COLORS:
            pose = list(self.positions[color])
            block = sim.getObject(f"/{color}_block")
            sensor = sim.getObject(f"/{color}_block_sensor")
            sim.setObjectParent(block, -1, True)
            sim.setObjectPose(block, -1, pose + [1, 0, 0, 0])
            self.blocks[color] = BlockInfo(block, sensor, tuple(pose))

        if self.config_mode == "mix":
            if not self.initial_stacks:
                self.initial_stacks = self._build_initial_stacks_from_areas(randomize=False)
            self._apply_initial_stacks(self.initial_stacks)
            self._refresh_block_positions()
        else:
            self.initial_stacks = []
        
        # Set human and serve points
        y_human = self.positions["human"][1]
        tray_pose = self.positions["tray"]
        y_tray = tray_pose[1]
        
        # serve_point_1 is on the tray, serve_point_2 is in front of human
        y_serve1 = self._get_serve_position_for_tray(y_tray)
        y_serve2 = self._get_serve_position_for_human(y_human)
        
        sim.setObjectPose(self.human, sim.handle_world, [3, y_human, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [3, y_human, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.human_goal, sim.handle_world, [1.1, y_human, 0, 0, 0, -1, 0])
        sim.setObjectPose(self.serve_1, sim.handle_world, [0.75, y_serve1, 0.88, 0, 0, 0, 1])
        sim.setObjectPose(self.serve_2, sim.handle_world, [0.75, y_serve2, 0.88, 0, 0, 0, 1])
        sim.setObjectPosition(self.tray, -1, tray_pose)
        sim.setObjectOrientation(self.tray, -1, self.tray_ori)
    
    def _reset_random(self) -> None:
        """Reset with random configuration."""
        x_coords = [0.4, 0.6]
        y_coords = [-0.65, -0.5, -0.35, -0.15, 0, 0.15, 0.35, 0.5, 0.65]
        z_coord = 0.83
        
        # Human position (randomized independently)
        y_human = random.choice([0.6, -0.6, 0])
        self._set_human_pose(y_human)
        self.areas["human"] = determine_area(y_human, threshold=0.1)
        time.sleep(1)
        
        # Tray position (randomized independently from human)
        y_tray = random.choice([0.45, -0.45, 0])
        # serve_point_1 is on the tray, serve_point_2 is in front of human
        y_serve1 = self._get_serve_position_for_tray(y_tray)
        y_serve2 = self._get_serve_position_for_human(y_human)
        
        sim.setObjectPosition(self.tray, -1, [0.8, y_tray, 0.84])
        sim.setObjectOrientation(self.tray, -1, self.tray_ori)
        self.positions["human"] = [1.1, y_human, 0]
        self.positions["tray"] = [0.8, y_tray, 0.84]
        self.areas["tray"] = determine_area(y_tray, threshold=0.3)
        
        sim.setObjectPose(self.serve_1, sim.handle_world, [0.75, y_serve1, 0.88, 0, 0, 0, 1])
        sim.setObjectPose(self.serve_2, sim.handle_world, [0.75, y_serve2, 0.88, 0, 0, 0, 1])
        
        # Random block positions
        all_positions = list(itertools.product(x_coords, y_coords))
        random.shuffle(all_positions)
        
        for color, (x, y) in zip(self.COLORS, all_positions[:len(self.COLORS)]):
            block = sim.getObject(f"/{color}_block")
            sensor = sim.getObject(f"/{color}_block_sensor")
            sim.setObjectParent(block, -1, True)
            sim.setObjectPose(block, -1, [x, y, z_coord, 1, 0, 0, 0])
            
            self.blocks[color] = BlockInfo(block, sensor, (x, y, z_coord))
            self.areas[color] = determine_area(y, threshold=0.25)
            self.positions[color] = (x, y, z_coord)

        if self.config_mode == "mix":
            self.initial_stacks = self._build_initial_stacks_from_areas(randomize=True)
            self._apply_initial_stacks(self.initial_stacks)
            self._refresh_block_positions()
        else:
            self.initial_stacks = []
        
        # Generate stacking goal
        selected_blocks = random.sample(self.COLORS, self.block_nums)
        stacking_choice = random.choice(self.stacking_options)
        self.type = stacking_choice
        
        random.shuffle(selected_blocks)
        self.stacks = []
        index = 0
        for size in stacking_choice:
            self.stacks.append(selected_blocks[index:index + size])
            index += size

    def _build_initial_stacks_from_areas(self, randomize: bool = True) -> List[List[str]]:
        """Create initial top-to-bottom stacks for blocks that share the same area."""
        area_groups: Dict[str, List[str]] = {"left": [], "overlap": [], "right": []}
        for color in self.COLORS:
            area = self.areas.get(color)
            if area in area_groups:
                area_groups[area].append(color)

        initial_stacks: List[List[str]] = []
        for area in ("left", "overlap", "right"):
            members = list(area_groups[area])
            if len(members) < 2:
                continue
            if randomize:
                random.shuffle(members)
            else:
                members.sort()
            initial_stacks.append(members)

        return initial_stacks

    def _apply_initial_stacks(self, initial_stacks: List[List[str]]) -> None:
        """Apply initial stack parent relationships using the same pattern as place()."""
        for top_to_bottom in initial_stacks:
            if len(top_to_bottom) < 2:
                continue

            bottom_to_top = list(reversed(top_to_bottom))
            for i in range(len(bottom_to_top) - 1):
                lower = bottom_to_top[i]
                upper = bottom_to_top[i + 1]

                upper_handle = self.blocks[upper].handle
                lower_handle = self.blocks[lower].handle

                sim.setObjectInt32Parameter(upper_handle, sim.shapeintparam_static, 1)
                sim.setObjectParent(upper_handle, -1, True)
                sim.setObjectParent(upper_handle, lower_handle, True)
                sim.setObjectPosition(upper_handle, lower_handle, self.INIT_STACK_RELATIVE_POS)

    def _refresh_block_positions(self) -> None:
        """Refresh cached block positions after initial stacking is applied."""
        for color in self.COLORS:
            handle = self.blocks[color].handle
            sensor = self.blocks[color].sensor
            position = tuple(sim.getObjectPosition(handle, -1))
            self.blocks[color] = BlockInfo(handle, sensor, position)
            self.positions[color] = position
    
    def _get_serve_position_for_tray(self, y_tray: float) -> float:
        """Get serve_point_1 y-coordinate based on tray position."""
        if y_tray > 0.3:
            return 0.4
        elif y_tray < -0.3:
            return -0.4
        return 0.1
    
    def _get_serve_position_for_human(self, y_human: float) -> float:
        """Get serve_point_2 y-coordinate based on human position."""
        if y_human > 0:
            return 0.35
        elif y_human < 0:
            return -0.35
        return -0.1
    
    def _format_stack(self, stack: List[str]) -> str:
        """Format stack description (bottom to top)."""
        return ", ".join(stack[::-1])

    def _format_direct_on_relations(self, top_to_bottom: List[str]) -> str:
        """Format direct support relations implied by a top-to-bottom order."""
        if len(top_to_bottom) <= 1:
            return "none (single-block pile)"
        pairs = [
            f"{top_to_bottom[i]} on {top_to_bottom[i + 1]}"
            for i in range(len(top_to_bottom) - 1)
        ]
        return "; ".join(pairs)

    def _build_pile_descriptions(self) -> List[Dict[str, Any]]:
        """Build pile descriptions used by both language goal and success check."""
        pile_desc = []
        for idx, stack in enumerate(self.stacks):
            serve_point = "serve_point_1" if idx == 0 else "serve_point_2"
            serve_sensor = self.serve_1_sensor if idx == 0 else self.serve_2_sensor
            pile_desc.append({
                "pile_id": idx + 1,
                "serve_point": serve_point,
                "serve_sensor": serve_sensor,
                "top_to_bottom": list(reversed(stack)),
            })
        return pile_desc
    
    def _generate_descriptions(self) -> None:
        """Generate task descriptions."""
        self.pile_desc = self._build_pile_descriptions()

        pile_count = len(self.pile_desc)

        pile_specs: List[str] = []
        for pile in self.pile_desc:
            top_to_bottom = pile["top_to_bottom"]
            serve_point = pile["serve_point"]
            if serve_point == "serve_point_1":
                serve_label = "serve_point_1 (on tray)"
            else:
                serve_label = "serve_point_2 (in front of human)"
            pile_specs.append(
                f"- Goal pile_{pile['pile_id']} -> {serve_label} goal_top_down: [{' on '.join(top_to_bottom)}]."
            )

        if pile_count == 1:
            pile_specs.append(
                "- serve_point_2 (in front of human) is available; no pile required."
            )

        pile_specs_text = "\n".join(pile_specs)

        self.task_des = f"""
### Task Information:
## Task Description: There are several blocks with different colors on the table. There are two serve points: serve_point_1 is on the tray on the table, and serve_point_2 is in front of the human (same area as the human).

## Task Goal:
Stack the blocks into {pile_count} pile(s). 

{pile_specs_text}

goal_top_down means [TOP on ... on BOTTOM]. The BOTTOM block of each pile must be at the serve_point. Only one block at one point.

## Task related objects: red_block, yellow_block, green_block, blue_block, orange_block, black_block
## Task related points: serve_point_1, serve_point_2, overlap_point_1, overlap_point_2
"""
        self.task_des_visual = self.describe_initial_state(
            human_area=self.areas['human'], block_areas={color: self.areas[color] for color in self.COLORS}
        )

    def describe_initial_state(self, human_area: str, block_areas: Dict[str, str]) -> str:
        """Scene description given to the agents; the areas come from the config or from OWLv2."""
        initial_visual = ""
        if self.config_mode == "mix" and self.initial_stacks:
            relations = [
                f"- Init stack_{idx}: init_top_down [{' on '.join(stack)}]."
                for idx, stack in enumerate(self.initial_stacks, start=1)
            ]
            initial_visual = " Initial stacks are: " + "; ".join(relations) + "."

        blocks = [f"the {color}_block is at the {block_areas[color]} area" for color in self.DESCRIPTION_ORDER]
        return f"""
## Initial State (from vision):

The human is standing in the {human_area} area, the tray is at the {self.areas['tray']} area, {', '.join(blocks[:-1])}, and {blocks[-1]}.{initial_visual} init_top_down means [TOP on ... on BOTTOM].
"""

    def check_success(self) -> bool:
        """Check if blocks satisfy the natural-language pile descriptions."""
        if not self.pile_desc:
            self.pile_desc = self._build_pile_descriptions()

        for pile in self.pile_desc:
            # Convert language order (top->bottom) to physics checks (bottom->top).
            bottom_to_top = list(reversed(pile["top_to_bottom"]))
            bottom = bottom_to_top[0]

            at_correct_serve = sim.checkProximitySensor(
                pile["serve_sensor"], self.blocks[bottom].handle
            )[0]
            if not at_correct_serve:
                print(
                    f"Stacking error: pile_{pile['pile_id']} expects {bottom} "
                    f"at {pile['serve_point']} as bottom block."
                )
                return False

            for i in range(len(bottom_to_top) - 1):
                lower, upper = bottom_to_top[i], bottom_to_top[i + 1]
                sensor = self.blocks[lower].sensor
                if not sim.checkProximitySensor(sensor, self.blocks[upper].handle)[0]:
                    print(f"Stacking error: the {upper} block is not detected over the {lower} block.")
                    return False
        
        return True
    
    def get_image_from_eyes(self, index: int):
        """Capture image (with extra delay for blocks)."""
        head.set_pose_target(NicolPose([0.75, 0.0, 1.1], [0, 0, 0, 0]))
        time.sleep(2)
        image = nicol.head_.get_left_eye_camera_img(False, prefix=f"{self.name}_{index}")
        head.set_pose_target(NicolPose([0.75, 0.0, 1.4], [0, 0, 0, 0]))
        return image


# =====================================================================
# FACTORY FUNCTION
# =====================================================================
def create_task(task_name: str, config: Optional[str] = None) -> Optional[BaseTask]:
    """
    Create a task instance by name.
    
    Args:
        task_name: One of "ServeWater", "ServeFruit", "StackBlock-simple", 
                   "StackBlock-complex", "StackBlock-super", "StackBlock"
        config: Optional configuration (used for StackBlock)
    
    Returns:
        Task instance or None if invalid name
    """
    task_map = {
        "ServeWater": lambda: ServeWaterTask(),
        "ServeFruit": lambda: ServeFruitTask(),
        "StackBlock-simple": lambda: StackBlockTask(config="simple"),
        "StackBlock-complex": lambda: StackBlockTask(config="complex"),
        "StackBlock-super": lambda: StackBlockTask(config="super"),
        "StackBlock": lambda: StackBlockTask(config="mix"),
    }
    
    if task_name in task_map:
        return task_map[task_name]()
    
    print(f"Unknown task: {task_name}")
    return None


def generate_visual_description(task: BaseTask, directions: Dict[str, str]) -> str:
    """
    Generate visual description based on detected object directions.
    
    Args:
        task: Task instance
        directions: Dictionary mapping object names to detected areas
    
    Returns:
        Formatted visual description string
    """
    if task.name == "ServeWater":
        return f"""
Based on the vision analyzation, the human is standing in the {directions['human face']} area, the blue cup is at the {directions['blue cup']} area, the yellow cup is at the {directions['yellow cup']} area, the brown box is at the {directions['brown box']} area, with the store_point in the central, and the water is inside the blue cup at the present.
"""
    
    elif task.name == "ServeFruit":
        return f"""
Based on the vision analyzation, the human is standing in the {directions['human face']} area, with the red bowl at the {directions['red bowl']} area, the apple at the {directions['apple']} area, and the banana at the {directions['banana']} area.
"""
    
    elif isinstance(task, StackBlockTask):
        # OWLv2 gives the areas of the human and the blocks ("center" is the overlap area). It
        # cannot see which blocks are stacked and does not detect the tray, so the initial stacks
        # and the tray area come from the scene, in the same format as the config description.
        def area(query: str) -> str:
            found = directions.get(query, "unknown")
            return "overlap" if found == "center" else found

        return task.describe_initial_state(
            human_area=area("human face"),
            block_areas={color: area(f"{color} block") for color in task.COLORS},
        )
    
    return ""
