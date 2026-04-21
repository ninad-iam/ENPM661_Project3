from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, PathJoinSubstitution, EnvironmentVariable, FindExecutable

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_share = FindPackageShare("diff_bot")

    world_file = PathJoinSubstitution([pkg_share, "worlds", "empty.world"])
    xacro_file = PathJoinSubstitution(
        [pkg_share, "urdf", "wheelchair_robot.urdf.xacro"])
    model_path = PathJoinSubstitution([pkg_share, "models"])

    gazebo_model_path = SetEnvironmentVariable(
        name="GAZEBO_MODEL_PATH",
        value=[
            model_path,
            ":",
            EnvironmentVariable("GAZEBO_MODEL_PATH", default_value="")
        ]
    )

    robot_description = ParameterValue(
        Command([
            FindExecutable(name="xacro"),
            " ",
            xacro_file
        ]),
        value_type=str
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare("gazebo_ros"),
                "launch",
                "gazebo.launch.py"
            ])
        ]),
        launch_arguments={
            "world": world_file
        }.items()
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": robot_description,
            "use_sim_time": True
        }]
    )

    # Spawn pose matched to the default A* start below
    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        arguments=[
            "-entity", "diff_bot",
            "-topic", "robot_description",
            "-x", "5.5",
            "-y", "0.0",
            "-z", "0.1",
            "-Y", "1.57"
        ],
        output="screen"
    )

    return LaunchDescription([
        gazebo_model_path,
        gazebo,
        robot_state_publisher,
        spawn_robot,
    ])
