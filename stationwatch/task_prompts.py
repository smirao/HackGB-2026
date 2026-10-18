"""Shared visible-behavior rules for the controller-and-machine demonstration."""

import re

CONTROLLER_RUBRIC = """For this assigned controller/joystick task, evaluate TWO visible requirements together:
1. CONTROLLER USE: a hand is on the joystick, buttons, or other operating controls,
   or visibly gripping the controller in an operating position. Judge hand placement;
   a still photo need not show the stick moving or a button being pressed. Merely
   standing beside the controller or touching its base is not sufficient evidence
   of operating it. Do not confuse a handheld phone with the task controller.
2. MACHINE ORIENTATION: the worker's head/face is broadly directed toward the
   assigned machine or its display. In the laptop demonstration, the open laptop
   is the machine/display to observe. Identify its position relative to the face;
   a side-profile view is sufficient. Do not require visible pupils, readable
   screen content, or proof that the laptop screen is illuminated. Torso direction
   alone does not establish that the face is directed toward the machine.

Choose on_task when BOTH controller use and machine orientation are visibly met.
Choose off_task when EITHER requirement is clearly violated. In particular:
- Hands operating the joystick BUT the head/face turned toward the camera, behind
  the worker, or otherwise clearly away from the laptop/machine is off_task.
- Hands visibly away from the controls, with the controller left untouched, is
  off_task even if the worker remains beside the station or faces the machine.
- Looking down at or using a separate phone instead of observing the machine is
  off_task, including when the other hand remains on the controller.
- Standing at the station, holding the controller, or facing the machine alone
  does not establish on_task; both required behaviors must be present.

"""

AMBIGUITY_RULES = """Choose uncertain for genuine missing visual evidence: the relevant hand/control
relationship or face/machine direction cannot be determined because of occlusion,
framing, blur, or ambiguous posture. However, a clearly visible violation of one
requirement is enough for off_task even when the other requirement is hidden.
Do not choose uncertain merely because the photo cannot prove motion, exact eye
fixation, duration, or the contents of the machine's screen. A head clearly turned
away is an off_task observation for THIS FRAME; the camera agent separately decides
whether such observations last long enough to warrant review. An ambiguous small
head turn is uncertain. Ignore unrelated background objects and room decor.
"""


def controller_task_rubric(task, binary=False):
    """Apply this workstation rule only to tasks that name a controller or joystick."""
    if not re.search(r"\b(?:joystick|controller)s?\b", task, re.IGNORECASE):
        return None
    if not binary:
        return CONTROLLER_RUBRIC + AMBIGUITY_RULES
    return CONTROLLER_RUBRIC + """Return only on_task or off_task. A clearly visible violation of either
requirement supports off_task even when the other requirement is hidden. When
evidence is incomplete, choose the label best supported by the visible posture
and report lower confidence; do not invent a violation solely because a hand is
partly hidden, motion is not visible, or the laptop screen content cannot be read.
No worker present at the station is off_task. If multiple people are visible,
assess the person operating or positioned at the assigned controller and machine.
A head clearly turned away is an off_task observation for THIS FRAME; the camera
agent separately decides whether these observations last long enough for review.
Ignore unrelated background objects and room decor.
"""
