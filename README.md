I have been working on a footstep planner for quadruped locomotion. As an initial task, I developed this repository that contains Python codes for generating feasible locomotion of a quadruped on discontinuous flat surfaces, staircases, and ramps. The surface data are hard-coded as of now; we expect to receive them using perception in the future. The results can be seen here: https://drive.google.com/drive/folders/16MGWFGTeUMIun8UVhOieIBaTqaC_qf8o?usp=sharing.

1. Clone 'go1_ros2' repository first and follow all the instructions as mentioned in https://github.com/ChiratheRobotics/go1_ros2/tree/dev.

2. In a new Tab/Window of the terminal, clone this repository
   
3. Run `rviz2' command in the terminal.
   
4. Change 'Fixed Frame' to 'base'.

5. Add RobotModel and change `Description Topic' to '/robot_description'.

6. The Go1 robot should be visible by now in the Rviz2 window.

7. Run the Python codes now as per your choice of terrain.
