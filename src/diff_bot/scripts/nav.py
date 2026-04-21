#!/usr/bin/env python3

import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


# ============================================================
# MAP IN WORLD COORDINATES
# ============================================================

def create_project3_map(clearance_cm: int,
                        x_min_m: float = 0.0,
                        x_max_m: float = 8.0,
                        y_min_m: float = -1.0,
                        y_max_m: float = 11.0) -> tuple[np.ndarray, dict]:
    """
    Creates a 2D occupancy map in world coordinates.
    Resolution: 1 cell = 1 cm
    For now: only outer boundary.
    """
    width_cm = int(round((x_max_m - x_min_m) * 100.0))
    height_cm = int(round((y_max_m - y_min_m) * 100.0))

    grid = np.zeros((height_cm, width_cm), dtype=np.uint8)
    c = clearance_cm

    map_info = {
        "x_min_m": x_min_m,
        "x_max_m": x_max_m,
        "y_min_m": y_min_m,
        "y_max_m": y_max_m,
        "width_cm": width_cm,
        "height_cm": height_cm,
    }

    # Outer boundary only
    grid[:, :c] = 1
    grid[:, width_cm - c:] = 1
    grid[:c, :] = 1
    grid[height_cm - c:, :] = 1

    return grid, map_info


def world_m_to_grid_cm(x_m: float, y_m: float, map_info: dict) -> Tuple[float, float]:
    x_cm = (x_m - map_info["x_min_m"]) * 100.0
    y_cm = (y_m - map_info["y_min_m"]) * 100.0
    return x_cm, y_cm


def grid_cm_to_world_m(x_cm: float, y_cm: float, map_info: dict) -> Tuple[float, float]:
    x_m = map_info["x_min_m"] + x_cm / 100.0
    y_m = map_info["y_min_m"] + y_cm / 100.0
    return x_m, y_m


def is_free(grid: np.ndarray, x_cm: float, y_cm: float) -> bool:
    xi = int(round(x_cm))
    yi = int(round(y_cm))
    if xi < 0 or yi < 0 or yi >= grid.shape[0] or xi >= grid.shape[1]:
        return False
    return grid[yi, xi] == 0


# ============================================================
# A* SEARCH
# ============================================================

@dataclass(order=True)
class PrioritizedItem:
    priority: float
    node: object


@dataclass
class SearchNode:
    x_cm: float
    y_cm: float
    theta_rad: float
    g_cost: float
    h_cost: float
    parent: Optional["SearchNode"]
    action: Optional[Tuple[float, float, float]]  # (v, w, dt)

    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost


def wrap_angle(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def heuristic(x_cm: float, y_cm: float, goal_x_cm: float, goal_y_cm: float) -> float:
    return math.hypot(goal_x_cm - x_cm, goal_y_cm - y_cm)


def quantize_state(x_cm: float,
                   y_cm: float,
                   theta_rad: float,
                   xy_res_cm: float = 5.0,
                   theta_res_deg: float = 15.0) -> Tuple[int, int, int]:
    qx = int(round(x_cm / xy_res_cm))
    qy = int(round(y_cm / xy_res_cm))
    qtheta = int(round(math.degrees(wrap_angle(theta_rad)) / theta_res_deg))
    return qx, qy, qtheta


def build_action_set(rpm1: float, rpm2: float) -> List[Tuple[float, float]]:
    return [
        (0.0, rpm1),
        (rpm1, 0.0),
        (rpm1, rpm1),
        (0.0, rpm2),
        (rpm2, 0.0),
        (rpm2, rpm2),
        (rpm1, rpm2),
        (rpm2, rpm1),
    ]


def rpm_pair_to_twist(rpm_l: float,
                      rpm_r: float,
                      wheel_radius_m: float,
                      wheel_base_m: float,
                      action_dt: float) -> Tuple[float, float, float]:
    wl = rpm_l * 2.0 * math.pi / 60.0
    wr = rpm_r * 2.0 * math.pi / 60.0
    v = wheel_radius_m * (wr + wl) / 2.0
    w = wheel_radius_m * (wr - wl) / wheel_base_m
    return v, w, action_dt


def simulate_action(x_cm: float,
                    y_cm: float,
                    theta_rad: float,
                    rpm_l: float,
                    rpm_r: float,
                    wheel_radius_m: float,
                    wheel_base_m: float,
                    action_dt: float,
                    grid: np.ndarray,
                    map_info: dict,
                    integration_dt: float = 0.1) -> Optional[Tuple[float, float, float, float]]:
    """
    Simulate one differential-drive action.
    Returns:
      new_x_cm, new_y_cm, new_theta_rad, travelled_distance_cm
    Returns None if collision occurs.
    """
    x_m, y_m = grid_cm_to_world_m(x_cm, y_cm, map_info)
    theta = theta_rad

    wl = rpm_l * 2.0 * math.pi / 60.0
    wr = rpm_r * 2.0 * math.pi / 60.0

    elapsed = 0.0
    travelled_m = 0.0

    while elapsed < action_dt:
        dt = min(integration_dt, action_dt - elapsed)

        v = wheel_radius_m * (wr + wl) / 2.0
        w = wheel_radius_m * (wr - wl) / wheel_base_m

        x_m += v * math.cos(theta) * dt
        y_m += v * math.sin(theta) * dt
        theta = wrap_angle(theta + w * dt)
        travelled_m += abs(v) * dt

        cx_cm, cy_cm = world_m_to_grid_cm(x_m, y_m, map_info)
        if not is_free(grid, cx_cm, cy_cm):
            return None

        elapsed += dt

    new_x_cm, new_y_cm = world_m_to_grid_cm(x_m, y_m, map_info)
    return new_x_cm, new_y_cm, theta, travelled_m * 100.0


def reconstruct_actions(goal_node: SearchNode) -> List[Tuple[float, float, float]]:
    actions = []
    current = goal_node
    while current.parent is not None and current.action is not None:
        actions.append(current.action)
        current = current.parent
    actions.reverse()
    return actions


def astar_plan(start_pose: Tuple[float, float, float],
               goal_xy: Tuple[float, float],
               grid: np.ndarray,
               map_info: dict,
               wheel_radius_m: float,
               wheel_base_m: float,
               rpm1: float,
               rpm2: float,
               action_dt: float,
               goal_threshold_m: float,
               logger) -> List[Tuple[float, float, float]]:

    start_x_cm, start_y_cm = world_m_to_grid_cm(
        start_pose[0], start_pose[1], map_info)
    goal_x_cm, goal_y_cm = world_m_to_grid_cm(goal_xy[0], goal_xy[1], map_info)
    goal_threshold_cm = goal_threshold_m * 100.0

    if not is_free(grid, start_x_cm, start_y_cm):
        raise RuntimeError(
            f"Start ({start_pose[0]:.2f}, {start_pose[1]:.2f}) is in obstacle/clearance"
        )
    if not is_free(grid, goal_x_cm, goal_y_cm):
        raise RuntimeError(
            f"Goal ({goal_xy[0]:.2f}, {goal_xy[1]:.2f}) is in obstacle/clearance"
        )

    start_node = SearchNode(
        x_cm=start_x_cm,
        y_cm=start_y_cm,
        theta_rad=start_pose[2],
        g_cost=0.0,
        h_cost=heuristic(start_x_cm, start_y_cm, goal_x_cm, goal_y_cm),
        parent=None,
        action=None
    )

    open_heap: List[PrioritizedItem] = []
    heapq.heappush(open_heap, PrioritizedItem(start_node.f_cost, start_node))

    best_cost: Dict[Tuple[int, int, int], float] = {
        quantize_state(start_node.x_cm, start_node.y_cm, start_node.theta_rad): 0.0
    }

    actions = build_action_set(rpm1, rpm2)
    expansions = 0
    max_expansions = 200000

    while open_heap:
        current = heapq.heappop(open_heap).node
        expansions += 1

        if expansions % 1000 == 0:
            logger.info(
                f"A*: expanded={expansions}, "
                f"pose=({current.x_cm:.1f}, {current.y_cm:.1f}, {math.degrees(current.theta_rad):.1f} deg), "
                f"f={current.f_cost:.2f}"
            )

        dist_goal = heuristic(current.x_cm, current.y_cm, goal_x_cm, goal_y_cm)
        if dist_goal <= goal_threshold_cm:
            logger.info(f"A*: goal reached after {expansions} expansions")
            return reconstruct_actions(current)

        if expansions > max_expansions:
            raise RuntimeError(
                "A*: max expansions exceeded. Adjust map, start/goal, or tune parameters.")

        for rpm_l, rpm_r in actions:
            result = simulate_action(
                current.x_cm,
                current.y_cm,
                current.theta_rad,
                rpm_l,
                rpm_r,
                wheel_radius_m,
                wheel_base_m,
                action_dt,
                grid,
                map_info
            )

            if result is None:
                continue

            new_x_cm, new_y_cm, new_theta, distance_cm = result
            new_g = current.g_cost + distance_cm
            key = quantize_state(new_x_cm, new_y_cm, new_theta)

            if key in best_cost and new_g >= best_cost[key]:
                continue

            best_cost[key] = new_g
            h = heuristic(new_x_cm, new_y_cm, goal_x_cm, goal_y_cm)

            child = SearchNode(
                x_cm=new_x_cm,
                y_cm=new_y_cm,
                theta_rad=new_theta,
                g_cost=new_g,
                h_cost=h,
                parent=current,
                action=rpm_pair_to_twist(
                    rpm_l, rpm_r, wheel_radius_m, wheel_base_m, action_dt)
            )

            heapq.heappush(open_heap, PrioritizedItem(child.f_cost, child))

    raise RuntimeError("A*: no path found")


# ============================================================
# ROS 2 EXECUTION NODE
# ============================================================

class AStarNavNode(Node):
    def __init__(self):
        super().__init__("astar_nav")

        self.declare_parameter("start", [5.5, 0.0, 1.57])
        self.declare_parameter("goal", [6.0, 10.0])

        # Use smaller planning footprint first so start is valid
        self.declare_parameter("robot_radius", 0.20)
        self.declare_parameter("clearance", 0.05)

        # From your Gazebo diff-drive log
        self.declare_parameter("wheel_radius", 0.30)
        self.declare_parameter("wheel_base", 0.69)

        self.declare_parameter("rpm1", 5.0)
        self.declare_parameter("rpm2", 10.0)
        self.declare_parameter("action_dt", 1.5)
        self.declare_parameter("goal_threshold", 0.20)

        start = self.get_parameter("start").value
        goal = self.get_parameter("goal").value
        robot_radius = float(self.get_parameter("robot_radius").value)
        clearance = float(self.get_parameter("clearance").value)
        wheel_radius = float(self.get_parameter("wheel_radius").value)
        wheel_base = float(self.get_parameter("wheel_base").value)
        rpm1 = float(self.get_parameter("rpm1").value)
        rpm2 = float(self.get_parameter("rpm2").value)
        action_dt = float(self.get_parameter("action_dt").value)
        goal_threshold = float(self.get_parameter("goal_threshold").value)

        total_clearance_cm = int(round((robot_radius + clearance) * 100.0))

        self.grid, self.map_info = create_project3_map(
            total_clearance_cm,
            x_min_m=0.0,
            x_max_m=8.0,
            y_min_m=-1.0,
            y_max_m=11.0
        )

        sx_cm, sy_cm = world_m_to_grid_cm(
            float(start[0]), float(start[1]), self.map_info)
        gx_cm, gy_cm = world_m_to_grid_cm(
            float(goal[0]), float(goal[1]), self.map_info)

        self.get_logger().info(
            f"Map bounds: x=[{self.map_info['x_min_m']}, {self.map_info['x_max_m']}], "
            f"y=[{self.map_info['y_min_m']}, {self.map_info['y_max_m']}]"
        )
        self.get_logger().info(
            f"Start world=({start[0]}, {start[1]}) -> grid=({sx_cm:.1f}, {sy_cm:.1f}) "
            f"free={is_free(self.grid, sx_cm, sy_cm)}"
        )
        self.get_logger().info(
            f"Goal world=({goal[0]}, {goal[1]}) -> grid=({gx_cm:.1f}, {gy_cm:.1f}) "
            f"free={is_free(self.grid, gx_cm, gy_cm)}"
        )

        self.publisher = self.create_publisher(Twist, "/cmd_vel", 10)

        self.actions = astar_plan(
            start_pose=(float(start[0]), float(start[1]), float(start[2])),
            goal_xy=(float(goal[0]), float(goal[1])),
            grid=self.grid,
            map_info=self.map_info,
            wheel_radius_m=wheel_radius,
            wheel_base_m=wheel_base,
            rpm1=rpm1,
            rpm2=rpm2,
            action_dt=action_dt,
            goal_threshold_m=goal_threshold,
            logger=self.get_logger()
        )

        self.get_logger().info(f"A*: planned {len(self.actions)} actions")

        self.current_action_index = 0
        self.action_start_time = None
        self.done = False

        self.timer = self.create_timer(0.05, self.on_timer)

    def publish_stop(self):
        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0
        self.publisher.publish(msg)

    def on_timer(self):
        if self.done:
            self.publish_stop()
            return

        if self.current_action_index >= len(self.actions):
            self.get_logger().info("A*: execution finished")
            self.publish_stop()
            self.done = True
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if self.action_start_time is None:
            self.action_start_time = now_sec
            v, w, dt = self.actions[self.current_action_index]
            self.get_logger().info(
                f"Executing step {self.current_action_index + 1}/{len(self.actions)}: "
                f"v={v:.3f} m/s, w={w:.3f} rad/s, dt={dt:.2f} s"
            )

        v, w, dt = self.actions[self.current_action_index]

        msg = Twist()
        msg.linear.x = float(v)
        msg.angular.z = float(w)
        self.publisher.publish(msg)

        if (now_sec - self.action_start_time) >= dt:
            self.current_action_index += 1
            self.action_start_time = None


def main():
    rclpy.init()
    node = AStarNavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
