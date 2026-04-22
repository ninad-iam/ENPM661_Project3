#!/usr/bin/env python3

import math
import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry



# SCALED FINAL MAP

def create_project3_map(clearance_cm: int, map_scale: float = 4.0):
    """
    Builds a 2D occupancy grid for the project 3 arena.

    The real arena is 2m x 4m. We scale it up by map_scale so
    the planner has more resolution to work with (e.g., 4x -> 8m x 16m).
    Everything—obstacles, walls, blocks—gets scaled the same way.

    Returns:
        grid            – 2D numpy array (0 = free, 1 = obstacle/inflated)
        map_info        – dict with bounds, dimensions, scale factor
        obstacle_shapes – list of obstacle dicts for plotting later
    """

    # Original arena dimensions (meters)
    base_x_max = 2.0
    base_y_max = 4.0

    # After scaling, these are the actual map bounds
    x_min_m, x_max_m = 0.0, base_x_max * map_scale
    y_min_m, y_max_m = 0.0, base_y_max * map_scale

    # Convert to centimeters for the grid (1 cell = 1 cm)
    width_cm = int(round((x_max_m - x_min_m) * 100.0))
    height_cm = int(round((y_max_m - y_min_m) * 100.0))

    # The occupancy grid — starts all free (zeros)
    grid = np.zeros((height_cm, width_cm), dtype=np.uint8)
    c = clearance_cm  # shorthand for inflation amount
    obstacle_shapes = []  # we'll collect shapes here for plotting

    def S(v):
        """Scale a base-map value to the enlarged map."""
        return v * map_scale

    def world_to_grid(x, y):
        """Convert world coordinates (meters) to grid indices (cm)."""
        return int(round(x * 100.0)), int(round(y * 100.0))

    def add_rect_raw(x1, y1, x2, y2):
        """
        Mark a rectangle as occupied on the grid (already in scaled meters).
        Inflates by clearance c on all sides, clamped to grid bounds.
        """
        gx1, gy1 = world_to_grid(x1, y1)
        gx2, gy2 = world_to_grid(x2, y2)

        xmin, xmax = sorted([gx1, gx2])
        ymin, ymax = sorted([gy1, gy2])

        grid[max(0, ymin - c):min(height_cm, ymax + c),
             max(0, xmin - c):min(width_cm, xmax + c)] = 1

    def add_rect(x1, y1, x2, y2, label=None):
        """
        Add a rectangle in BASE (unscaled) coordinates.
        Scales it up, marks the grid, and saves it for plotting.
        """
        x1s, y1s, x2s, y2s = S(x1), S(y1), S(x2), S(y2)
        add_rect_raw(x1s, y1s, x2s, y2s)

        obstacle_shapes.append({
            "type": "rect",
            "x_min": min(x1s, x2s),
            "x_max": max(x1s, x2s),
            "y_min": min(y1s, y2s),
            "y_max": max(y1s, y2s),
            "label": label,
        })

    def add_thick_line(x1, y1, length, angle_deg, thickness=0.05, label=None):
        """
        Draw an angled wall by stamping small rectangles along the line.
        Brute-force but works for arbitrary angles without polygon math.
        """
        x1s = S(x1)
        y1s = S(y1)
        length_s = S(length)
        thickness_s = S(thickness)

        # Compute the endpoint from start + angle + length
        theta = math.radians(angle_deg)
        x2s = x1s + length_s * math.cos(theta)
        y2s = y1s + length_s * math.sin(theta)

        # Walk along the line, stamping small squares at each step
        steps = max(120, int(120 * map_scale))
        for i in range(steps + 1):
            t = i / steps
            x = x1s + t * (x2s - x1s)
            y = y1s + t * (y2s - y1s)
            add_rect_raw(
                x - thickness_s / 2.0,
                y - thickness_s / 2.0,
                x + thickness_s / 2.0,
                y + thickness_s / 2.0
            )

        obstacle_shapes.append({
            "type": "line",
            "x1": x1s, "y1": y1s,
            "x2": x2s, "y2": y2s,
            "thickness": thickness_s,
            "label": label,
        })

    #  Outer walls (thin rectangles along the arena boundary)      
    add_rect(0.0, 0.0, 2.0, 0.05, "bottom wall")
    add_rect(0.0, 3.95, 2.0, 4.0, "top wall")
    add_rect(0.0, 0.0, 0.05, 4.0, "left wall")
    add_rect(1.95, 0.0, 2.0, 4.0, "right wall")

    # 3 square blocks (0.30m x 0.30m each, centered at given positions)      
    block_size = 0.30
    half = block_size / 2.0
    add_rect(1.55 - half, 0.42 - half, 1.55 + half, 0.42 + half, "block_1")
    add_rect(0.50 - half, 1.50 - half, 0.50 + half, 1.50 + half, "block_2")
    add_rect(0.30 - half, 2.70 - half, 0.30 + half, 2.70 + half, "block_3")

    #  Internal walls (angled lines that partition the space)      
    add_thick_line(
        x1=0.10, y1=0.40,
        length=1.41, angle_deg=30,
        thickness=0.05, label="30 deg wall"
    )
    add_thick_line(
        x1=1.9, y1=1.40,
        length=1.25, angle_deg=155,
        thickness=0.05, label="155 deg wall"
    )
    add_thick_line(
        x1=0.1, y1=2.92,
        length=1.45, angle_deg=0,
        thickness=0.05, label="0 deg wall"
    )

    map_info = {
        "x_min_m": x_min_m,
        "x_max_m": x_max_m,
        "y_min_m": y_min_m,
        "y_max_m": y_max_m,
        "width_cm": width_cm,
        "height_cm": height_cm,
        "map_scale": map_scale,
    }

    return grid, map_info, obstacle_shapes


def world_m_to_grid_cm(x_m: float, y_m: float, map_info: dict):
    """World meters -> grid cm indices (rounded to nearest cell)."""
    x_cm = int(round((x_m - map_info["x_min_m"]) * 100.0))
    y_cm = int(round((y_m - map_info["y_min_m"]) * 100.0))
    return x_cm, y_cm


def grid_cm_to_world_m(x_cm: int, y_cm: int, map_info: dict):
    """Grid cm indices -> world meters."""
    x_m = map_info["x_min_m"] + x_cm / 100.0
    y_m = map_info["y_min_m"] + y_cm / 100.0
    return x_m, y_m


def is_free(grid, x_cm, y_cm):
    """Check if a grid cell is in bounds and not occupied."""
    if x_cm < 0 or y_cm < 0 or y_cm >= grid.shape[0] or x_cm >= grid.shape[1]:
        return False
    return grid[y_cm, x_cm] == 0


# GRID A* 
# Standard A* on the cm-resolution occupancy grid.
# Uses 8-connected neighbors (cardinal + diagonal).


@dataclass(order=True)
class PQItem:
    """Wrapper so we can shove GridNodes into a min-heap by f-cost."""
    priority: float
    count: int       # tiebreaker — older nodes pop first (FIFO on ties)
    node: object


@dataclass
class GridNode:
    """One cell in the search graph."""
    x: int
    y: int
    g: float                         # cost so far from start
    h: float                         # heuristic estimate to goal
    parent: Optional["GridNode"]     # backpointer for path reconstruction

    @property
    def f(self) -> float:
        """Total estimated cost (the thing we sort the heap on)."""
        return self.g + self.h


def heuristic(x: int, y: int, gx: int, gy: int) -> float:
    """Euclidean distance heuristic — admissible for 8-connected grid."""
    return math.hypot(gx - x, gy - y)


def get_neighbors(x: int, y: int):
    """
    Yield all 8 neighbors with their step costs.
    Cardinal moves cost 1.0, diagonals cost sqrt(2).
    """
    moves = [
        (-1,  0, 1.0), (1,  0, 1.0),
        (0, -1, 1.0), (0,  1, 1.0),
        (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),  (1,  1, math.sqrt(2)),
    ]
    for dx, dy, cost in moves:
        yield x + dx, y + dy, cost


def reconstruct_path(node: GridNode) -> List[Tuple[int, int]]:
    """Walk the parent chain back to start and reverse it."""
    path = []
    current = node
    while current is not None:
        path.append((current.x, current.y))
        current = current.parent
    path.reverse()
    return path


def astar_grid(grid: np.ndarray,
               start_xy: Tuple[int, int],
               goal_xy: Tuple[int, int],
               logger):
    """
    Run A* on the occupancy grid.

    Returns:
        path     – list of (x_cm, y_cm) from start to goal
        explored – all cells we popped (for visualization)

    Raises RuntimeError if start/goal are blocked or no path exists.
    """
    sx, sy = start_xy
    gx, gy = goal_xy

    # Sanity check — don't even bother if start or goal is inside an obstacle
    if not is_free(grid, sx, sy):
        raise RuntimeError(f"Start grid=({sx}, {sy}) is in obstacle")
    if not is_free(grid, gx, gy):
        raise RuntimeError(f"Goal grid=({gx}, {gy}) is in obstacle")

    start = GridNode(
        x=sx, y=sy,
        g=0.0,
        h=heuristic(sx, sy, gx, gy),
        parent=None
    )

    # Min-heap ordered by f-cost, with a counter for stable tie-breaking
    open_heap: List[PQItem] = []
    push_count = 0
    heapq.heappush(open_heap, PQItem(start.f, push_count, start))
    push_count += 1

    # Track the best g-cost seen for each cell so we can skip stale entries
    best_cost: Dict[Tuple[int, int], float] = {(sx, sy): 0.0}

    # Every cell we pop — saved so we can draw the exploration wavefront
    explored: List[Tuple[int, int]] = []

    while open_heap:
        current = heapq.heappop(open_heap).node
        explored.append((current.x, current.y))

        # Are we there yet?
        if (current.x, current.y) == (gx, gy):
            logger.info(f"A*: goal reached, explored {len(explored)} cells")
            return reconstruct_path(current), explored

        # Expand all 8 neighbors
        for nx, ny, step_cost in get_neighbors(current.x, current.y):
            if not is_free(grid, nx, ny):
                continue

            new_g = current.g + step_cost
            key = (nx, ny)

            # Skip if we already found a cheaper way to this cell
            if key in best_cost and new_g >= best_cost[key]:
                continue

            best_cost[key] = new_g
            child = GridNode(
                x=nx, y=ny,
                g=new_g,
                h=heuristic(nx, ny, gx, gy),
                parent=current
            )
            heapq.heappush(open_heap, PQItem(child.f, push_count, child))
            push_count += 1

    raise RuntimeError("A*: no path found")


# PATH UTILITIES


def downsample_path_world(path_xy_cm: List[Tuple[int, int]],
                          map_info: dict,
                          spacing_m: float = 0.25) -> List[Tuple[float, float]]:
    """
    Take the dense cm-level A* path and thin it out to waypoints
    spaced roughly `spacing_m` apart (in world meters).

    The robot follower doesn't need a waypoint every centimeter —
    it just needs enough to stay on course through the corridors.
    Always includes the first and last points.
    """
    if not path_xy_cm:
        return []

    waypoints = []
    last_x_m, last_y_m = grid_cm_to_world_m(
        path_xy_cm[0][0], path_xy_cm[0][1], map_info)
    waypoints.append((last_x_m, last_y_m))

    for x_cm, y_cm in path_xy_cm[1:]:
        x_m, y_m = grid_cm_to_world_m(x_cm, y_cm, map_info)
        # Only keep this point if we've traveled far enough from the last one
        if math.hypot(x_m - last_x_m, y_m - last_y_m) >= spacing_m:
            waypoints.append((x_m, y_m))
            last_x_m, last_y_m = x_m, y_m

    # Make sure the actual goal is always the last waypoint
    end_x_m, end_y_m = grid_cm_to_world_m(
        path_xy_cm[-1][0], path_xy_cm[-1][1], map_info)
    if not waypoints or math.hypot(end_x_m - waypoints[-1][0], end_y_m - waypoints[-1][1]) > 1e-6:
        waypoints.append((end_x_m, end_y_m))

    return waypoints


def wrap_angle(a: float) -> float:
    """Wrap an angle to [-pi, pi] the lazy but correct way."""
    return math.atan2(math.sin(a), math.cos(a))


def yaw_from_quat(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw from a quaternion (only works if roll/pitch are ~0)."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)



# LIVE PLOT


def setup_live_plot(grid, map_info, obstacle_shapes, explored, path, start=None, goal=None):
    """
    Set up the matplotlib figure that shows the map, A* exploration,
    planned path, and a live dot for the robot's odometry.

    Returns handles we need to update the robot position in real time.
    """
    plt.ion()  # interactive mode so the plot doesn't block
    fig, ax = plt.subplots(figsize=(8, 14))

    # Show the occupancy grid as a grayscale image
    extent = [
        map_info["x_min_m"],
        map_info["x_max_m"],
        map_info["y_min_m"],
        map_info["y_max_m"],
    ]

    ax.imshow(
        grid,
        origin="lower",
        cmap="gray_r",
        extent=extent,
        interpolation="nearest",
        alpha=0.85
    )

    # Draw obstacle outlines on top
    for obs in obstacle_shapes:
        if obs["type"] == "rect":
            rect = Rectangle(
                (obs["x_min"], obs["y_min"]),
                obs["x_max"] - obs["x_min"],
                obs["y_max"] - obs["y_min"],
                fill=False,
                edgecolor="cyan",
                linewidth=1.2
            )
            ax.add_patch(rect)
        elif obs["type"] == "line":
            ax.plot(
                [obs["x1"], obs["x2"]],
                [obs["y1"], obs["y2"]],
                color="magenta",
                linewidth=2.5
            )

    # Show every cell A* explored (tiny orange dots — gives you the wavefront)
    if explored:
        ex, ey = [], []
        for x_cm, y_cm in explored:
            x_m, y_m = grid_cm_to_world_m(x_cm, y_cm, map_info)
            ex.append(x_m)
            ey.append(y_m)
        ax.plot(ex, ey, '.', color='orange',
                markersize=0.7, alpha=0.2, label="Explored")

    # The planned path (blue line)
    if path:
        px, py = [], []
        for x_cm, y_cm in path:
            x_m, y_m = grid_cm_to_world_m(x_cm, y_cm, map_info)
            px.append(x_m)
            py.append(y_m)
        ax.plot(px, py, 'b-', linewidth=2.0, label="A* Path")

    # Start and goal markers
    if start is not None:
        ax.plot(start[0], start[1], 'go', markersize=8, label="Start")
    if goal is not None:
        ax.plot(goal[0], goal[1], 'ro', markersize=8, label="Goal")

    # These two will be updated in real time as odometry comes in
    robot_dot, = ax.plot([], [], 'mo', markersize=8, label="Robot")
    robot_trail, = ax.plot([], [], 'm--', linewidth=1.2,
                           alpha=0.8, label="Odom Trail")

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Scaled A* Path + Live Robot Odom (scale={map_info['map_scale']:.1f}x)")
    ax.grid(True)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    return fig, ax, robot_dot, robot_trail
  

class AStarDriveNode(Node):
    def __init__(self):
        super().__init__("astar_drive")

        # Declare all ROS parameters with defaults 
        # These can be overridden from the launch file or command line

        # Map scaling (4x means 2x4m -> 8x16m)
        self.declare_parameter("map_scale", 4.0)

        # Start and goal in the ORIGINAL 2x4m frame (will be scaled)
        self.declare_parameter("base_start", [1.0, 0.30])
        self.declare_parameter("base_goal", [1.0, 3.5])

        # How much to inflate obstacles (robot footprint + safety margin)
        self.declare_parameter("robot_radius", 0.325)
        self.declare_parameter("clearance", 0.02)

        # Waypoint follower tuning knobs
        self.declare_parameter("plot_only", False)       # if True, just show the plan, don't drive
        self.declare_parameter("waypoint_spacing", 0.25) # meters between waypoints
        self.declare_parameter("waypoint_tol", 0.15)     # how close is "close enough" to a waypoint
        self.declare_parameter("yaw_tol", 0.30)          # rad — if yaw error > this, rotate in place
        self.declare_parameter("k_lin", 0.5)             # proportional gain for linear velocity
        self.declare_parameter("k_ang", 0.8)             # proportional gain for angular velocity
        self.declare_parameter("max_lin", 0.25)           # m/s speed cap
        self.declare_parameter("max_ang", 0.8)            # rad/s turn rate cap

        #  Read them all back 
        map_scale = float(self.get_parameter("map_scale").value)
        base_start = self.get_parameter("base_start").value
        base_goal = self.get_parameter("base_goal").value
        robot_radius = float(self.get_parameter("robot_radius").value)
        clearance = float(self.get_parameter("clearance").value)

        self.plot_only = bool(self.get_parameter("plot_only").value)
        self.waypoint_spacing = float(self.get_parameter("waypoint_spacing").value)
        self.waypoint_tol = float(self.get_parameter("waypoint_tol").value)
        self.yaw_tol = float(self.get_parameter("yaw_tol").value)
        self.k_lin = float(self.get_parameter("k_lin").value)
        self.k_ang = float(self.get_parameter("k_ang").value)
        self.max_lin = float(self.get_parameter("max_lin").value)
        self.max_ang = float(self.get_parameter("max_ang").value)

        # Total inflation = robot's physical radius + a little extra
        total_clearance_cm = int(round((robot_radius + clearance) * 100.0))

        #  Build the map and run A*

        self.grid, self.map_info, self.obstacle_shapes = create_project3_map(
            total_clearance_cm,
            map_scale=map_scale
        )

        # Scale start and goal to match the enlarged map
        self.start = (float(base_start[0]) * map_scale,
                      float(base_start[1]) * map_scale)
        self.goal = (float(base_goal[0]) * map_scale,
                     float(base_goal[1]) * map_scale)

        # Convert to grid coords and check they're in free space
        sx_cm, sy_cm = world_m_to_grid_cm(
            self.start[0], self.start[1], self.map_info)
        gx_cm, gy_cm = world_m_to_grid_cm(
            self.goal[0], self.goal[1], self.map_info)

        self.get_logger().info(
            f"Scaled start={self.start} -> grid=({sx_cm}, {sy_cm}) "
            f"free={is_free(self.grid, sx_cm, sy_cm)}"
        )
        self.get_logger().info(
            f"Scaled goal={self.goal} -> grid=({gx_cm}, {gy_cm}) "
            f"free={is_free(self.grid, gx_cm, gy_cm)}"
        )

        # Run the planner
        path_cm, explored = astar_grid(
            self.grid,
            (sx_cm, sy_cm),
            (gx_cm, gy_cm),
            self.get_logger()
        )

        # Thin out the dense path into manageable waypoints
        self.waypoints = downsample_path_world(
            path_cm,
            self.map_info,
            self.waypoint_spacing
        )
        self.get_logger().info(
            f"A*: {len(path_cm)} raw path cells, {len(self.waypoints)} waypoints"
        )

        #   Set up the live plot

        self.fig, self.ax, self.robot_dot, self.robot_trail = setup_live_plot(
            self.grid,
            self.map_info,
            self.obstacle_shapes,
            explored,
            path_cm,
            start=self.start,
            goal=self.goal
        )

        #  ROS pub/sub

        # Listen to odometry so we know where the robot actually is
        self.odom_sub = self.create_subscription(
            Odometry, "/odom", self.odom_cb, 10)

        # Publish velocity commands to drive the robot
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Robot state — will be filled in by the odom callback
        self.robot_x = None
        self.robot_y = None
        self.robot_yaw = None

        # Which waypoint we're heading toward right now
        self.wp_index = 0
        self.done = self.plot_only  # skip driving if plot_only is set

        # For drawing the odometry trail on the plot
        self.odom_x_hist: List[float] = []
        self.odom_y_hist: List[float] = []

        # Throttle debug prints (every 10th timer tick)
        self.debug_counter = 0

        if self.plot_only:
            self.get_logger().info("Plot-only mode enabled. Robot will not move.")
        else:
            self.get_logger().info("Waypoint following enabled.")
            # 20 Hz control loop
            self.timer = self.create_timer(0.05, self.on_timer)

    #     ROS callbacks and control logic

    def odom_cb(self, msg: Odometry):
        """Grab the robot's current position and heading from odometry."""
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = yaw_from_quat(q.x, q.y, q.z, q.w)

    def publish_stop(self):
        """Send zero velocity (stop the robot)."""
        msg = Twist()
        self.cmd_pub.publish(msg)

    def update_live_plot(self):
        """
        Move the robot dot and trail on the matplotlib figure.
        Called every control tick so the plot stays in sync with reality.
        """
        if self.robot_x is None or self.robot_y is None:
            return

        self.odom_x_hist.append(self.robot_x)
        self.odom_y_hist.append(self.robot_y)

        # Cap the trail length so we don't eat all the RAM on long runs
        if len(self.odom_x_hist) > 5000:
            self.odom_x_hist = self.odom_x_hist[-5000:]
            self.odom_y_hist = self.odom_y_hist[-5000:]

        self.robot_dot.set_data([self.robot_x], [self.robot_y])
        self.robot_trail.set_data(self.odom_x_hist, self.odom_y_hist)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    #   main control loop

    def on_timer(self):
        """
        Runs at 20 Hz. The logic is simple:
          1. If we're done, just stop and keep updating the plot.
          2. If we're close enough to the current waypoint, move to the next.
          3. If the heading error is too big, rotate in place first.
          4. Otherwise, drive forward while correcting heading.

        It's a basic "turn then go" controller — not fancy, but
        good enough for following a pre-planned path in simulation.
        """
        if self.done:
            self.publish_stop()
            self.update_live_plot()
            return

        # Wait until we have at least one odom reading
        if self.robot_x is None or self.robot_y is None or self.robot_yaw is None:
            return

        self.update_live_plot()

        # All waypoints done — stop the robot and exit
        if self.wp_index >= len(self.waypoints):
            self.get_logger().info("Reached final waypoint. Stopping.")
            self.publish_stop()
            self.done = True
            return

        # Current target waypoint
        tx, ty = self.waypoints[self.wp_index]
        dx = tx - self.robot_x
        dy = ty - self.robot_y
        dist = math.hypot(dx, dy)

        # What direction do we need to face?
        target_yaw = math.atan2(dy, dx)
        yaw_error = wrap_angle(target_yaw - self.robot_yaw)

        # Print status every 10 ticks (~2 Hz) so the log isn't overwhelming
        self.debug_counter += 1
        if self.debug_counter % 10 == 0:
            self.get_logger().info(
                f"pose=({self.robot_x:.2f}, {self.robot_y:.2f}, "
                f"{math.degrees(self.robot_yaw):.1f} deg) | "
                f"wp={self.wp_index+1}/{len(self.waypoints)} "
                f"target=({tx:.2f}, {ty:.2f}) | "
                f"dist={dist:.2f} yaw_err={math.degrees(yaw_error):.1f} deg"
            )

        # Close enough to this waypoint? Advance to the next one.
        if dist < self.waypoint_tol:
            self.get_logger().info(
                f"Reached waypoint {self.wp_index+1}/{len(self.waypoints)} "
                f"at ({tx:.2f}, {ty:.2f})"
            )
            self.wp_index += 1
            return

        cmd = Twist()

        if abs(yaw_error) > self.yaw_tol:
            # Heading is way off — stop and rotate in place first
            cmd.linear.x = 0.0
            cmd.angular.z = max(-self.max_ang,
                                min(self.max_ang, self.k_ang * yaw_error))
        else:
            # Heading is roughly right — drive forward with steering correction
            cmd.linear.x = min(self.max_lin, self.k_lin * dist)
            cmd.angular.z = max(-self.max_ang,
                                min(self.max_ang, self.k_ang * yaw_error))

        self.cmd_pub.publish(cmd)


def main():
    rclpy.init()
    node = AStarDriveNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()