# This message describes a 2-finger grasp configuration by its 6-DOF pose
# expressed wrt to the world reference frame, and the opening
# width of the robot hand.

geometry_msgs/PoseStamped pose # grasp pose

std_msgs/Float32 width # Required aperture (opening width) of the robot hand

std_msgs/Float32 score # Score assigned to the grasp by the classifier
