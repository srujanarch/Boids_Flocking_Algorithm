
"""
====================================================================
ADVANCED BOIDS - THREE DRONE SWARM SIMULATION
====================================================================

Simulation only.

Features
--------
1. Separation
2. Predictive separation
3. Alignment
4. Cohesion
5. Individual goal navigation
6. Adaptive goal weighting
7. Triangle formation control
8. Formation relaxation near obstacles
9. 3D spherical obstacle avoidance
10. Predictive obstacle avoidance
11. Speed limiting
12. Acceleration limiting
13. Smooth braking near goals
14. Stuck detection
15. Stuck recovery
16. Goal completion
17. Collision statistics
18. Path-length statistics
19. 3D trajectory visualization
20. Animated GIF

Requirements
------------
numpy
matplotlib
pillow

Install:
    pip install numpy matplotlib pillow

Run:
    python3 advanced_boids_three_drones.py

Output:
    advanced_boids_three_drones.gif
====================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter


# ==================================================================
# GLOBAL CONFIGURATION
# ==================================================================

DT = 0.05

MAX_TIME = 45.0

MAX_STEPS = int(MAX_TIME / DT)

NUM_DRONES = 3


# ==================================================================
# DRONE PARAMETERS
# ==================================================================

DRONE_RADIUS = 0.35

MAX_SPEED = 2.6

MAX_ACCELERATION = 2.2

MAX_DECELERATION = 2.8


# ==================================================================
# BOIDS PARAMETERS
# ==================================================================

NEIGHBOR_RADIUS = 7.0

SEPARATION_DISTANCE = 2.2

FORMATION_DISTANCE = 3.0


# ==================================================================
# BASE BOIDS WEIGHTS
# ==================================================================

SEPARATION_WEIGHT = 3.5

PREDICTIVE_SEPARATION_WEIGHT = 4.0

ALIGNMENT_WEIGHT = 0.8

COHESION_WEIGHT = 0.45

FORMATION_WEIGHT = 0.8

GOAL_WEIGHT_MIN = 1.5

GOAL_WEIGHT_MAX = 4.5

OBSTACLE_WEIGHT = 5.0


# ==================================================================
# OBSTACLE PARAMETERS
# ==================================================================

OBSTACLE_INFLUENCE = 3.5

PREDICTIVE_OBSTACLE_TIME = 1.5


# ==================================================================
# GOAL PARAMETERS
# ==================================================================

GOAL_TOLERANCE = 0.45

GOAL_SLOWDOWN_DISTANCE = 3.5

FINAL_GOAL_WEIGHT = 8.0


# ==================================================================
# STUCK DETECTION
# ==================================================================

STUCK_TIME = 2.0

STUCK_DISTANCE = 0.35

STUCK_RECOVERY_TIME = 1.5


# ==================================================================
# UTILITY FUNCTIONS
# ==================================================================

def norm(v):
    return np.linalg.norm(v)


def normalize(v):

    n = norm(v)

    if n < 1e-12:
        return np.zeros_like(v)

    return v / n


def limit_vector(v, maximum):

    n = norm(v)

    if n < 1e-12:
        return np.zeros_like(v)

    if n > maximum:
        return v / n * maximum

    return v


def clamp(value, minimum, maximum):

    return max(minimum, min(value, maximum))


def distance(a, b):

    return norm(a - b)


# ==================================================================
# SPHERICAL OBSTACLE
# ==================================================================

class SphereObstacle:

    def __init__(self, center, radius):

        self.center = np.asarray(
            center,
            dtype=float
        )

        self.radius = radius


# ==================================================================
# DRONE
# ==================================================================

class Drone:

    def __init__(
        self,
        drone_id,
        position,
        goal,
        velocity=None
    ):

        self.id = drone_id

        self.position = np.asarray(
            position,
            dtype=float
        )

        self.goal = np.asarray(
            goal,
            dtype=float
        )

        if velocity is None:

            self.velocity = np.zeros(3)

        else:

            self.velocity = np.asarray(
                velocity,
                dtype=float
            )

        self.radius = DRONE_RADIUS

        self.reached = False

        self.reach_time = None

        self.recovery_timer = 0.0

        self.stuck_timer = 0.0

        self.previous_position = self.position.copy()

        self.path_length = 0.0

        self.history = [
            self.position.copy()
        ]

        self.velocity_history = [
            self.velocity.copy()
        ]

        self.acceleration_history = []

    # --------------------------------------------------------------
    # Goal information
    # --------------------------------------------------------------

    def goal_vector(self):

        return self.goal - self.position


    def goal_distance(self):

        return norm(
            self.goal_vector()
        )


    def goal_direction(self):

        return normalize(
            self.goal_vector()
        )


    # --------------------------------------------------------------
    # Desired velocity toward goal
    # --------------------------------------------------------------

    def desired_goal_velocity(self):

        d = self.goal_distance()

        if d < GOAL_TOLERANCE:

            return np.zeros(3)

        direction = self.goal_direction()

        speed = MAX_SPEED

        if d < GOAL_SLOWDOWN_DISTANCE:

            speed = MAX_SPEED * (
                d / GOAL_SLOWDOWN_DISTANCE
            )

            speed = max(
                speed,
                0.25
            )

        return direction * speed


    # --------------------------------------------------------------
    # Goal check
    # --------------------------------------------------------------

    def update_goal_status(self, current_time):

        d = self.goal_distance()

        if not self.reached and d < GOAL_TOLERANCE:

            self.reached = True

            self.reach_time = current_time

            self.velocity[:] = 0.0

            return True

        return False


# ==================================================================
# NEIGHBOR SEARCH
# ==================================================================

def get_neighbors(drone, drones):

    neighbors = []

    for other in drones:

        if other.id == drone.id:
            continue

        d = distance(
            drone.position,
            other.position
        )

        if d < NEIGHBOR_RADIUS:

            neighbors.append(
                (other, d)
            )

    return neighbors


# ==================================================================
# SEPARATION
# ==================================================================

def separation(drone, drones):

    force = np.zeros(3)

    neighbors = get_neighbors(
        drone,
        drones
    )

    for other, d in neighbors:

        if d < 1e-8:

            continue

        if d < SEPARATION_DISTANCE:

            direction = (
                drone.position
                - other.position
            )

            direction = normalize(
                direction
            )

            strength = (
                SEPARATION_DISTANCE - d
            ) / SEPARATION_DISTANCE

            strength = strength ** 2

            force += direction * strength

    return force


# ==================================================================
# PREDICTIVE SEPARATION
# ==================================================================

def predictive_separation(
    drone,
    drones
):

    force = np.zeros(3)

    prediction_time = 1.5

    predicted_self = (
        drone.position
        + drone.velocity
        * prediction_time
    )

    for other in drones:

        if other.id == drone.id:
            continue

        predicted_other = (
            other.position
            + other.velocity
            * prediction_time
        )

        relative = (
            predicted_self
            - predicted_other
        )

        d = norm(relative)

        danger_distance = (
            SEPARATION_DISTANCE
            + 0.8
        )

        if d < danger_distance:

            if d < 1e-8:

                direction = np.array(
                    [1.0, 0.0, 0.0]
                )

            else:

                direction = relative / d

            strength = (
                danger_distance - d
            ) / danger_distance

            force += (
                direction
                * strength
                * strength
            )

    return force


# ==================================================================
# ALIGNMENT
# ==================================================================

def alignment(drone, drones):

    neighbors = get_neighbors(
        drone,
        drones
    )

    if not neighbors:

        return np.zeros(3)

    average_velocity = np.zeros(3)

    for other, _ in neighbors:

        average_velocity += (
            other.velocity
        )

    average_velocity /= len(neighbors)

    return (
        average_velocity
        - drone.velocity
    )


# ==================================================================
# COHESION
# ==================================================================

def cohesion(drone, drones):

    neighbors = get_neighbors(
        drone,
        drones
    )

    if not neighbors:

        return np.zeros(3)

    center = np.zeros(3)

    for other, _ in neighbors:

        center += other.position

    center /= len(neighbors)

    return (
        center
        - drone.position
    )


# ==================================================================
# ADAPTIVE TRIANGLE FORMATION
# ==================================================================

def formation_control(
    drone,
    drones,
    obstacle_danger
):

    """
    Creates a soft triangular formation.

    Formation control is relaxed when:
        - a drone is close to its goal
        - an obstacle is nearby
        - the swarm needs to separate
    """

    active = [
        d for d in drones
        if d.id != drone.id
        and not d.reached
    ]

    if len(active) < 2:

        return np.zeros(3)

    # --------------------------------------------------------------
    # Determine swarm center
    # --------------------------------------------------------------

    center = np.mean(
        [
            d.position
            for d in drones
            if not d.reached
        ],
        axis=0
    )

    # --------------------------------------------------------------
    # Direction of swarm travel
    # --------------------------------------------------------------

    goal_center = np.mean(
        [
            d.goal
            for d in drones
            if not d.reached
        ],
        axis=0
    )

    travel_direction = normalize(
        goal_center - center
    )

    if norm(travel_direction) < 1e-8:

        travel_direction = np.array(
            [1.0, 0.0, 0.0]
        )

    # --------------------------------------------------------------
    # Construct lateral direction
    # --------------------------------------------------------------

    up = np.array(
        [0.0, 0.0, 1.0]
    )

    lateral = np.cross(
        travel_direction,
        up
    )

    lateral = normalize(
        lateral
    )

    if norm(lateral) < 1e-8:

        lateral = np.array(
            [1.0, 0.0, 0.0]
        )

    # --------------------------------------------------------------
    # Target formation position
    # --------------------------------------------------------------

    if drone.id == 0:

        offset = np.zeros(3)

    elif drone.id == 1:

        offset = (
            lateral * FORMATION_DISTANCE
        )

    else:

        offset = (
            -lateral
            * FORMATION_DISTANCE
        )

    target = center + offset

    error = target - drone.position

    # --------------------------------------------------------------
    # Relax formation when obstacles are nearby
    # --------------------------------------------------------------

    relaxation = 1.0 - obstacle_danger

    relaxation = clamp(
        relaxation,
        0.0,
        1.0
    )

    return (
        error
        * relaxation
        * 0.4
    )


# ==================================================================
# OBSTACLE AVOIDANCE
# ==================================================================

def obstacle_avoidance(
    drone,
    obstacles
):

    force = np.zeros(3)

    danger = 0.0

    for obstacle in obstacles:

        relative = (
            drone.position
            - obstacle.center
        )

        d = norm(relative)

        safe_radius = (
            obstacle.radius
            + drone.radius
        )

        influence_radius = (
            safe_radius
            + OBSTACLE_INFLUENCE
        )

        if d < influence_radius:

            if d < 1e-8:

                direction = np.array(
                    [1.0, 0.0, 0.0]
                )

            else:

                direction = (
                    relative / d
                )

            strength = (
                influence_radius - d
            ) / influence_radius

            # Strong nonlinear repulsion
            strength = strength ** 2

            force += (
                direction
                * strength
            )

            danger = max(
                danger,
                strength
            )

        # ----------------------------------------------------------
        # Predictive obstacle avoidance
        # ----------------------------------------------------------

        predicted = (
            drone.position
            + drone.velocity
            * PREDICTIVE_OBSTACLE_TIME
        )

        predicted_relative = (
            predicted
            - obstacle.center
        )

        predicted_distance = norm(
            predicted_relative
        )

        if predicted_distance < influence_radius:

            direction = normalize(
                predicted_relative
            )

            strength = (
                influence_radius
                - predicted_distance
            ) / influence_radius

            force += (
                direction
                * strength
                * 1.5
            )

            danger = max(
                danger,
                strength
            )

    return force, danger


# ==================================================================
# DYNAMIC GOAL WEIGHT
# ==================================================================

def dynamic_goal_weight(drone):

    d = drone.goal_distance()

    if d < GOAL_TOLERANCE:

        return 0.0

    if d < GOAL_SLOWDOWN_DISTANCE:

        # Goal dominates near target
        return FINAL_GOAL_WEIGHT

    ratio = clamp(
        d / 10.0,
        0.0,
        1.0
    )

    return (
        GOAL_WEIGHT_MIN
        + (
            GOAL_WEIGHT_MAX
            - GOAL_WEIGHT_MIN
        )
        * ratio
    )


# ==================================================================
# STUCK DETECTION
# ==================================================================

def update_stuck_state(drone):

    movement = distance(
        drone.position,
        drone.previous_position
    )

    speed = norm(
        drone.velocity
    )

    if movement < STUCK_DISTANCE * DT:

        if speed < 0.4:

            drone.stuck_timer += DT

    else:

        drone.stuck_timer = max(
            0.0,
            drone.stuck_timer - DT
        )

    drone.previous_position = (
        drone.position.copy()
    )

    if drone.stuck_timer > STUCK_TIME:

        drone.recovery_timer = (
            STUCK_RECOVERY_TIME
        )

        drone.stuck_timer = 0.0


# ==================================================================
# STUCK RECOVERY
# ==================================================================

def recovery_force(drone, drones):

    if drone.recovery_timer <= 0.0:

        return np.zeros(3)

    # Find closest drone
    closest = None

    closest_distance = np.inf

    for other in drones:

        if other.id == drone.id:
            continue

        d = distance(
            drone.position,
            other.position
        )

        if d < closest_distance:

            closest_distance = d

            closest = other

    # --------------------------------------------------------------
    # Lateral escape direction
    # --------------------------------------------------------------

    if closest is not None:

        away = (
            drone.position
            - closest.position
        )

        away = normalize(
            away
        )

    else:

        away = np.array(
            [1.0, 0.0, 0.0]
        )

    # Add vertical component
    escape = (
        away
        + np.array(
            [0.0, 0.0, 0.8]
        )
    )

    return normalize(
        escape
    )


# ==================================================================
# COMPLETE STEERING CALCULATION
# ==================================================================

def calculate_steering(
    drone,
    drones,
    obstacles
):

    # --------------------------------------------------------------
    # If goal reached, stop
    # --------------------------------------------------------------

    if drone.reached:

        return -drone.velocity * 2.0

    # --------------------------------------------------------------
    # Goal
    # --------------------------------------------------------------

    desired_goal_velocity = (
        drone.desired_goal_velocity()
    )

    goal_force = (
        desired_goal_velocity
        - drone.velocity
    )

    goal_weight = dynamic_goal_weight(
        drone
    )

    # --------------------------------------------------------------
    # Separation
    # --------------------------------------------------------------

    sep = separation(
        drone,
        drones
    )

    # --------------------------------------------------------------
    # Predictive separation
    # --------------------------------------------------------------

    pred_sep = predictive_separation(
        drone,
        drones
    )

    # --------------------------------------------------------------
    # Alignment
    # --------------------------------------------------------------

    align = alignment(
        drone,
        drones
    )

    # --------------------------------------------------------------
    # Cohesion
    # --------------------------------------------------------------

    coh = cohesion(
        drone,
        drones
    )

    # --------------------------------------------------------------
    # Obstacle avoidance
    # --------------------------------------------------------------

    obstacle, obstacle_danger = (
        obstacle_avoidance(
            drone,
            obstacles
        )
    )

    # --------------------------------------------------------------
    # Formation
    # --------------------------------------------------------------

    formation = formation_control(
        drone,
        drones,
        obstacle_danger
    )

    # --------------------------------------------------------------
    # Stuck recovery
    # --------------------------------------------------------------

    recovery = recovery_force(
        drone,
        drones
    )

    # --------------------------------------------------------------
    # Combine
    # --------------------------------------------------------------

    steering = (

        SEPARATION_WEIGHT
        * sep

        + PREDICTIVE_SEPARATION_WEIGHT
        * pred_sep

        + ALIGNMENT_WEIGHT
        * align

        + COHESION_WEIGHT
        * coh

        + FORMATION_WEIGHT
        * formation

        + goal_weight
        * goal_force

        + OBSTACLE_WEIGHT
        * obstacle

        + 3.0
        * recovery
    )

    # --------------------------------------------------------------
    # Emergency separation
    # --------------------------------------------------------------

    for other in drones:

        if other.id == drone.id:
            continue

        d = distance(
            drone.position,
            other.position
        )

        collision_distance = (
            drone.radius
            + other.radius
        )

        emergency_distance = (
            collision_distance
            + 0.35
        )

        if d < emergency_distance:

            direction = normalize(
                drone.position
                - other.position
            )

            if norm(direction) < 1e-8:

                direction = np.array(
                    [1.0, 0.0, 0.0]
                )

            emergency_strength = (
                emergency_distance - d
            ) / emergency_distance

            steering += (
                direction
                * emergency_strength
                * 10.0
            )

    # --------------------------------------------------------------
    # Limit acceleration
    # --------------------------------------------------------------

    steering = limit_vector(
        steering,
        MAX_ACCELERATION
    )

    return steering


# ==================================================================
# SIMULATION STEP
# ==================================================================

def simulation_step(
    drones,
    obstacles,
    current_time
):

    new_velocities = []

    # --------------------------------------------------------------
    # Compute all steering first
    # --------------------------------------------------------------

    steerings = []

    for drone in drones:

        update_stuck_state(
            drone
        )

        steering = calculate_steering(
            drone,
            drones,
            obstacles
        )

        steerings.append(
            steering
        )

    # --------------------------------------------------------------
    # Update velocity
    # --------------------------------------------------------------

    for drone, steering in zip(
        drones,
        steerings
    ):

        if drone.reached:

            new_velocities.append(
                np.zeros(3)
            )

            continue

        # ----------------------------------------------------------
        # Desired acceleration
        # ----------------------------------------------------------

        acceleration = steering

        # ----------------------------------------------------------
        # Additional braking near goal
        # ----------------------------------------------------------

        d = drone.goal_distance()

        if d < GOAL_SLOWDOWN_DISTANCE:

            velocity_toward_goal = (
                np.dot(
                    drone.velocity,
                    drone.goal_direction()
                )
            )

            if velocity_toward_goal > 0:

                stopping_distance = (
                    velocity_toward_goal ** 2
                    / (
                        2
                        * MAX_DECELERATION
                    )
                )

                if stopping_distance > d:

                    acceleration -= (
                        drone.goal_direction()
                        * MAX_DECELERATION
                    )

        acceleration = limit_vector(
            acceleration,
            MAX_ACCELERATION
        )

        # ----------------------------------------------------------
        # Integrate velocity
        # ----------------------------------------------------------

        new_velocity = (
            drone.velocity
            + acceleration * DT
        )

        # ----------------------------------------------------------
        # Limit speed
        # ----------------------------------------------------------

        new_velocity = limit_vector(
            new_velocity,
            MAX_SPEED
        )

        new_velocities.append(
            new_velocity
        )

    # --------------------------------------------------------------
    # Apply updates simultaneously
    # --------------------------------------------------------------

    for drone, new_velocity in zip(
        drones,
        new_velocities
    ):

        old_position = (
            drone.position.copy()
        )

        acceleration = (
            new_velocity
            - drone.velocity
        ) / DT

        drone.velocity = new_velocity

        drone.position += (
            drone.velocity
            * DT
        )

        # ----------------------------------------------------------
        # Path length
        # ----------------------------------------------------------

        drone.path_length += distance(
            drone.position,
            old_position
        )

        # ----------------------------------------------------------
        # Histories
        # ----------------------------------------------------------

        drone.history.append(
            drone.position.copy()
        )

        drone.velocity_history.append(
            drone.velocity.copy()
        )

        drone.acceleration_history.append(
            acceleration.copy()
        )

        # ----------------------------------------------------------
        # Recovery timer
        # ----------------------------------------------------------

        drone.recovery_timer = max(
            0.0,
            drone.recovery_timer - DT
        )

        # ----------------------------------------------------------
        # Goal
        # ----------------------------------------------------------

        drone.update_goal_status(
            current_time
        )


# ==================================================================
# COLLISION METRICS
# ==================================================================

def calculate_clearance(drones):

    minimum = np.inf

    for i in range(len(drones)):

        for j in range(i + 1, len(drones)):

            d = distance(
                drones[i].position,
                drones[j].position
            )

            clearance = (
                d
                - drones[i].radius
                - drones[j].radius
            )

            minimum = min(
                minimum,
                clearance
            )

    return minimum


# ==================================================================
# OBSTACLE COLLISION CHECK
# ==================================================================

def calculate_obstacle_clearance(
    drones,
    obstacles
):

    minimum = np.inf

    for drone in drones:

        for obstacle in obstacles:

            d = distance(
                drone.position,
                obstacle.center
            )

            clearance = (
                d
                - obstacle.radius
                - drone.radius
            )

            minimum = min(
                minimum,
                clearance
            )

    return minimum


# ==================================================================
# CREATE SCENARIO
# ==================================================================

def create_scenario():

    """
    Three drones have crossing trajectories.

    The center obstacle forces the swarm to adapt.
    """

    drones = [

        Drone(
            drone_id=0,
            position=[
                -8.0,
                -5.0,
                2.5
            ],
            goal=[
                8.0,
                5.0,
                4.0
            ],
            velocity=[
                0.8,
                0.5,
                0.0
            ]
        ),

        Drone(
            drone_id=1,
            position=[
                0.0,
                -8.0,
                3.0
            ],
            goal=[
                0.0,
                8.0,
                4.5
            ],
            velocity=[
                0.0,
                0.8,
                0.0
            ]
        ),

        Drone(
            drone_id=2,
            position=[
                8.0,
                -5.0,
                2.0
            ],
            goal=[
                -8.0,
                5.0,
                3.5
            ],
            velocity=[
                -0.8,
                0.5,
                0.0
            ]
        )
    ]

    return drones


# ==================================================================
# CREATE OBSTACLES
# ==================================================================

def create_obstacles():

    obstacles = [

        SphereObstacle(
            center=[
                0.0,
                0.0,
                3.0
            ],
            radius=1.5
        ),

        SphereObstacle(
            center=[
                -4.0,
                1.5,
                3.0
            ],
            radius=1.1
        ),

        SphereObstacle(
            center=[
                4.0,
                1.5,
                3.0
            ],
            radius=1.1
        ),

        SphereObstacle(
            center=[
                0.0,
                5.0,
                4.5
            ],
            radius=1.0
        )
    ]

    return obstacles


# ==================================================================
# RUN SIMULATION
# ==================================================================

def run_simulation(
    drones,
    obstacles
):

    print()
    print("=" * 72)
    print("ADVANCED BOIDS THREE-DRONE SWARM SIMULATION")
    print("=" * 72)

    print()
    print("Configuration")
    print("-" * 72)

    print(
        f"Number of drones       : "
        f"{NUM_DRONES}"
    )

    print(
        f"Time step              : "
        f"{DT:.2f} s"
    )

    print(
        f"Maximum simulation     : "
        f"{MAX_TIME:.1f} s"
    )

    print(
        f"Maximum speed          : "
        f"{MAX_SPEED:.2f} m/s"
    )

    print(
        f"Maximum acceleration   : "
        f"{MAX_ACCELERATION:.2f} m/s²"
    )

    print(
        f"Neighbor radius        : "
        f"{NEIGHBOR_RADIUS:.2f} m"
    )

    print(
        f"Separation distance    : "
        f"{SEPARATION_DISTANCE:.2f} m"
    )

    print()
    print("Drone configuration")
    print("-" * 72)

    for drone in drones:

        print()
        print(
            f"Drone {drone.id}"
        )

        print(
            f"    Start = "
            f"{drone.position}"
        )

        print(
            f"    Goal  = "
            f"{drone.goal}"
        )

    print()
    print("Starting simulation...")
    print()

    minimum_drone_clearance = np.inf

    minimum_obstacle_clearance = np.inf

    collision_samples = 0

    obstacle_collision_samples = 0

    first_goal_time = None

    all_goal_time = None

    for step in range(MAX_STEPS):

        current_time = step * DT

        # ----------------------------------------------------------
        # Metrics before update
        # ----------------------------------------------------------

        drone_clearance = calculate_clearance(
            drones
        )

        obstacle_clearance = (
            calculate_obstacle_clearance(
                drones,
                obstacles
            )
        )

        minimum_drone_clearance = min(
            minimum_drone_clearance,
            drone_clearance
        )

        minimum_obstacle_clearance = min(
            minimum_obstacle_clearance,
            obstacle_clearance
        )

        if drone_clearance < 0:

            collision_samples += 1

        if obstacle_clearance < 0:

            obstacle_collision_samples += 1

        # ----------------------------------------------------------
        # Check goals
        # ----------------------------------------------------------

        reached_count = sum(
            drone.reached
            for drone in drones
        )

        if (
            first_goal_time is None
            and reached_count > 0
        ):

            first_goal_time = current_time

        if reached_count == len(drones):

            all_goal_time = current_time

            print(
                f"All drones reached their goals "
                f"at t = {current_time:.2f} s"
            )

            break

        # ----------------------------------------------------------
        # Step
        # ----------------------------------------------------------

        simulation_step(
            drones,
            obstacles,
            current_time
        )

        # ----------------------------------------------------------
        # Progress
        # ----------------------------------------------------------

        if step % 100 == 0:

            distances = [
                drone.goal_distance()
                for drone in drones
            ]

            print(
                f"t={current_time:6.2f}s | "
                f"goal distances: "
                f"{distances[0]:5.2f}, "
                f"{distances[1]:5.2f}, "
                f"{distances[2]:5.2f} m"
            )

    else:

        print()
        print(
            "Maximum simulation time reached."
        )

    # ==================================================================
    # RESULTS
    # ==================================================================

    print()
    print("=" * 72)
    print("SIMULATION RESULTS")
    print("=" * 72)

    for drone in drones:

        goal_distance = (
            drone.goal_distance()
        )

        final_speed = norm(
            drone.velocity
        )

        speeds = [
            norm(v)
            for v in drone.velocity_history
        ]

        max_recorded_speed = max(
            speeds
        )

        print()
        print(
            f"Drone {drone.id}:"
        )

        print(
            f"    Final position = "
            f"{drone.position}"
        )

        print(
            f"    Goal           = "
            f"{drone.goal}"
        )

        print(
            f"    Goal distance  = "
            f"{goal_distance:.4f} m"
        )

        print(
            f"    Final speed    = "
            f"{final_speed:.4f} m/s"
        )

        print(
            f"    Maximum speed  = "
            f"{max_recorded_speed:.4f} m/s"
        )

        print(
            f"    Path length    = "
            f"{drone.path_length:.3f} m"
        )

        if drone.reached:

            print(
                f"    Status         = "
                f"GOAL REACHED"
            )

            print(
                f"    Reach time     = "
                f"{drone.reach_time:.2f} s"
            )

        else:

            print(
                f"    Status         = "
                f"GOAL NOT REACHED"
            )

    print()
    print(
        f"Minimum drone clearance     : "
        f"{minimum_drone_clearance:.4f} m"
    )

    print(
        f"Minimum obstacle clearance  : "
        f"{minimum_obstacle_clearance:.4f} m"
    )

    print()

    if minimum_drone_clearance >= 0:

        print(
            "DRONE COLLISION CHECK       : PASS"
        )

    else:

        print(
            "DRONE COLLISION CHECK       : FAIL"
        )

        print(
            f"Collision samples           : "
            f"{collision_samples}"
        )

    if minimum_obstacle_clearance >= 0:

        print(
            "OBSTACLE COLLISION CHECK    : PASS"
        )

    else:

        print(
            "OBSTACLE COLLISION CHECK    : FAIL"
        )

        print(
            f"Obstacle collision samples  : "
            f"{obstacle_collision_samples}"
        )

    print()

    if all_goal_time is not None:

        print(
            f"ALL GOALS REACHED           : "
            f"YES ({all_goal_time:.2f} s)"
        )

    else:

        print(
            "ALL GOALS REACHED           : NO"
        )

    print()

    if first_goal_time is not None:

        print(
            f"FIRST GOAL REACHED          : "
            f"{first_goal_time:.2f} s"
        )

    else:

        print(
            "FIRST GOAL REACHED          : NO"
        )

    print("=" * 72)

    return drones


# ==================================================================
# 3D ANIMATION
# ==================================================================

def animate_simulation(
    drones,
    obstacles,
    output_file="advanced_boids_three_drones.gif",
    fps=20
):

    print()
    print("=" * 72)
    print("CREATING 3D ANIMATION")
    print("=" * 72)

    histories = [
        np.array(
            drone.history
        )
        for drone in drones
    ]

    max_len = max(
        len(h)
        for h in histories
    )

    # --------------------------------------------------------------
    # Pad histories
    # --------------------------------------------------------------

    padded = []

    for history in histories:

        if len(history) < max_len:

            last = history[-1]

            extra = np.repeat(
                last.reshape(1, 3),
                max_len - len(history),
                axis=0
            )

            history = np.vstack(
                [history, extra]
            )

        padded.append(history)

    # --------------------------------------------------------------
    # Figure
    # --------------------------------------------------------------

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    all_positions = np.vstack(
        padded
    )

    margin = 2.0

    ax.set_xlim(
        all_positions[:, 0].min() - margin,
        all_positions[:, 0].max() + margin
    )

    ax.set_ylim(
        all_positions[:, 1].min() - margin,
        all_positions[:, 1].max() + margin
    )

    ax.set_zlim(
        max(
            0,
            all_positions[:, 2].min() - margin
        ),
        all_positions[:, 2].max() + margin
    )

    ax.set_xlabel(
        "X (m)"
    )

    ax.set_ylabel(
        "Y (m)"
    )

    ax.set_zlabel(
        "Z (m)"
    )

    ax.set_title(
        "Advanced Boids Three-Drone Swarm"
    )

    colors = [
        "red",
        "blue",
        "green"
    ]

    # --------------------------------------------------------------
    # Trails
    # --------------------------------------------------------------

    trails = []

    points = []

    for i, drone in enumerate(drones):

        trail, = ax.plot(
            [],
            [],
            [],
            "-",
            color=colors[i],
            linewidth=2.2,
            alpha=0.8
        )

        point, = ax.plot(
            [],
            [],
            [],
            "o",
            color=colors[i],
            markersize=9
        )

        trails.append(
            trail
        )

        points.append(
            point
        )

        # Goal marker
        ax.scatter(
            drone.goal[0],
            drone.goal[1],
            drone.goal[2],
            color=colors[i],
            marker="X",
            s=130,
            label=f"Drone {i} goal"
        )

    # --------------------------------------------------------------
    # Obstacles
    # --------------------------------------------------------------

    for obstacle in obstacles:

        u = np.linspace(
            0,
            2 * np.pi,
            24
        )

        v = np.linspace(
            0,
            np.pi,
            16
        )

        x = (
            obstacle.radius
            * np.outer(
                np.cos(u),
                np.sin(v)
            )
            + obstacle.center[0]
        )

        y = (
            obstacle.radius
            * np.outer(
                np.sin(u),
                np.sin(v)
            )
            + obstacle.center[1]
        )

        z = (
            obstacle.radius
            * np.outer(
                np.ones_like(u),
                np.cos(v)
            )
            + obstacle.center[2]
        )

        ax.plot_surface(
            x,
            y,
            z,
            alpha=0.25
        )

    # --------------------------------------------------------------
    # Text
    # --------------------------------------------------------------

    text = ax.text2D(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10
    )

    # --------------------------------------------------------------
    # Update
    # --------------------------------------------------------------

    def update(frame):

        for i, history in enumerate(
            padded
        ):

            positions = history[
                :frame + 1
            ]

            trails[i].set_data(
                positions[:, 0],
                positions[:, 1]
            )

            trails[i].set_3d_properties(
                positions[:, 2]
            )

            current = history[
                frame
            ]

            points[i].set_data(
                [current[0]],
                [current[1]]
            )

            points[i].set_3d_properties(
                [current[2]]
            )

        current_positions = np.array(
            [
                h[frame]
                for h in padded
            ]
        )

        center = np.mean(
            current_positions,
            axis=0
        )

        clearance = np.inf

        for i in range(
            len(current_positions)
        ):

            for j in range(
                i + 1,
                len(current_positions)
            ):

                d = distance(
                    current_positions[i],
                    current_positions[j]
                )

                c = (
                    d
                    - 2 * DRONE_RADIUS
                )

                clearance = min(
                    clearance,
                    c
                )

        text.set_text(
            f"Time: {frame * DT:.2f} s\n"
            f"Swarm center: "
            f"({center[0]:.2f}, "
            f"{center[1]:.2f}, "
            f"{center[2]:.2f})\n"
            f"Minimum clearance: "
            f"{clearance:.2f} m"
        )

        return (
            trails
            + points
            + [text]
        )

    # --------------------------------------------------------------
    # Animation
    # --------------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=max_len,
        interval=1000 / fps,
        blit=False
    )

    print(
        f"Saving animation: "
        f"{output_file}"
    )

    animation.save(
        output_file,
        writer=PillowWriter(
            fps=fps
        )
    )

    plt.close(fig)

    print(
        "Animation saved successfully."
    )


# ==================================================================
# STATIC TRAJECTORY PLOT
# ==================================================================

def plot_trajectories(
    drones,
    obstacles
):

    fig = plt.figure(
        figsize=(11, 9)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    colors = [
        "red",
        "blue",
        "green"
    ]

    for i, drone in enumerate(
        drones
    ):

        history = np.array(
            drone.history
        )

        ax.plot(
            history[:, 0],
            history[:, 1],
            history[:, 2],
            color=colors[i],
            linewidth=2,
            label=f"Drone {i}"
        )

        # Start
        ax.scatter(
            history[0, 0],
            history[0, 1],
            history[0, 2],
            color=colors[i],
            marker="o",
            s=70
        )

        # Goal
        ax.scatter(
            drone.goal[0],
            drone.goal[1],
            drone.goal[2],
            color=colors[i],
            marker="X",
            s=120
        )

    # --------------------------------------------------------------
    # Obstacles
    # --------------------------------------------------------------

    for obstacle in obstacles:

        u = np.linspace(
            0,
            2 * np.pi,
            24
        )

        v = np.linspace(
            0,
            np.pi,
            16
        )

        x = (
            obstacle.radius
            * np.outer(
                np.cos(u),
                np.sin(v)
            )
            + obstacle.center[0]
        )

        y = (
            obstacle.radius
            * np.outer(
                np.sin(u),
                np.sin(v)
            )
            + obstacle.center[1]
        )

        z = (
            obstacle.radius
            * np.outer(
                np.ones_like(u),
                np.cos(v)
            )
            + obstacle.center[2]
        )

        ax.plot_surface(
            x,
            y,
            z,
            alpha=0.2
        )

    ax.set_xlabel(
        "X (m)"
    )

    ax.set_ylabel(
        "Y (m)"
    )

    ax.set_zlabel(
        "Z (m)"
    )

    ax.set_title(
        "Advanced Three-Drone Boids Trajectories"
    )

    ax.legend()

    plt.tight_layout()

    plt.show()


# ==================================================================
# MAIN
# ==================================================================

if __name__ == "__main__":

    print()
    print(
        "Initializing advanced Boids swarm..."
    )

    drones = create_scenario()

    obstacles = create_obstacles()

    drones = run_simulation(
        drones,
        obstacles
    )

    animate_simulation(
        drones,
        obstacles,
        output_file=(
            "advanced_boids_three_drones.gif"
        ),
        fps=20
    )

    plot_trajectories(
        drones,
        obstacles
    )

    print()
    print("=" * 72)
    print("ADVANCED BOIDS SIMULATION COMPLETE")
    print("=" * 72)
    print()
    print(
        "Generated:"
    )
    print(
        "    advanced_boids_three_drones.gif"
    )
    print()


