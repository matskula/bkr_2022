import math
from dataclasses import dataclass
from typing import Tuple, Dict, Optional, Union
from pathlib import Path
from copy import copy
import glob
import shutil
import time

import cv2

from database import Database

DIST_THRESHOLD = 44
LINE_THRESHOLD = 20


class SpeedDetector:

    @dataclass
    class MovingObject:
        start_line_frame: Optional[Union[int, float]]
        end_line_frame: Optional[Union[int, float]]
        last_position: Tuple[int, int]   # cx cy

    def __init__(
        self,
        start_line_y: int,
        end_line_y: int,
        zone_length_m: int,
        speed_limit: int,
        fps: Optional[int] = None
    ):
        self._start_line_y: int = start_line_y
        self._end_line_y: int = end_line_y
        self._id_to_moving_objects: Dict[int, SpeedDetector.MovingObject] = {}
        self._id_count: int = 1
        self._frame: Union[int, float] = 0
        self._fps = fps
        self._zone_length_m = zone_length_m
        self._speed_limit = speed_limit
        self._db = Database()

    def update(self, objects_rect, img):

        if self._fps:
            self._frame += 1
        else:
            self._frame = time.time()

        updated_ids = set()

        for rect in objects_rect:
            x, y, w, h, index = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            updated_id = None

            for moving_object_id, moving_object in self._id_to_moving_objects.items():
                dist = math.hypot(cx - moving_object.last_position[0], cy - moving_object.last_position[1])

                if dist < DIST_THRESHOLD:

                    if cy - LINE_THRESHOLD < self._end_line_y < cy and not moving_object.end_line_frame:
                        moving_object.end_line_frame = self._frame

                    if cy - LINE_THRESHOLD < self._start_line_y < cy and not moving_object.start_line_frame:
                        moving_object.start_line_frame = self._frame

                    moving_object.last_position = (cx, cy)
                    updated_id = moving_object_id

                    Path(f"vehicles/{moving_object_id}").mkdir(parents=True, exist_ok=True)
                    our_img = copy(img)
                    cv2.circle(our_img, moving_object.last_position, 2, (0, 0, 255), -1)
                    cv2.imwrite(f"vehicles/{moving_object_id}/{self._frame}.jpg", our_img)
                    break

            else:
                if not (self._end_line_y > cy > self._start_line_y):
                    self._id_to_moving_objects[self._id_count] = SpeedDetector.MovingObject(
                        start_line_frame=None,
                        end_line_frame=None,
                        last_position=(cx, cy)
                    )
                    updated_id = self._id_count
                    self._id_count += 1
            if updated_id:
                updated_ids.add(updated_id)

        to_delete_ids = []

        for moving_object_id in self._id_to_moving_objects.keys():
            if moving_object_id not in updated_ids:
                to_delete_ids.append(moving_object_id)

        for to_delete_id in to_delete_ids:
            moving_object = self._id_to_moving_objects.pop(to_delete_id)
            if moving_object.start_line_frame and moving_object.end_line_frame:

                frames = abs(moving_object.start_line_frame - moving_object.end_line_frame)
                if self._fps:
                    time_passed = frames * (1 / self._fps)
                else:
                    time_passed = frames
                speed_m_s = self._zone_length_m / time_passed
                speed_km_h = speed_m_s * 3.6
                print(speed_km_h)
                if speed_km_h > self._speed_limit:
                    file_path = f'vehicles/{to_delete_id}/proof_{int(speed_km_h)}.avi'

                    out = cv2.VideoWriter(
                        file_path,
                        cv2.VideoWriter_fourcc(*'I420'), self._fps, (img.shape[1], img.shape[0])
                    )

                    for filename in sorted(glob.glob(f'vehicles/{to_delete_id}/*.jpg')):
                        img = cv2.imread(filename)
                        out.write(img)

                    out.release()

                    with self._db.connection as cur:
                        cur.execute(f"""
                            insert into driver(speed, path)
                            values ({speed_km_h}, '{file_path}')
    
                        """)
                else:
                    shutil.rmtree(f'vehicles/{to_delete_id}', ignore_errors=True)
