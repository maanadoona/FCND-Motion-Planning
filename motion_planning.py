import argparse
import time
import msgpack
from enum import Enum, auto

import numpy as np

#from planning_utils import a_star, heuristic, create_grid, collinearity_prune, plot_graph_skeleton, create_grid_and_edges, plot_graph_network, closest_point
import planning_utils as pu
from udacidrone import Drone
from udacidrone.connection import MavlinkConnection
from udacidrone.messaging import MsgID
from udacidrone.frame_utils import global_to_local

import csv

from skimage.morphology import medial_axis
from skimage.util import invert
import numpy.linalg as LA
import networkx as nx


class States(Enum):
    MANUAL = auto()
    ARMING = auto()
    TAKEOFF = auto()
    WAYPOINT = auto()
    LANDING = auto()
    DISARMING = auto()
    PLANNING = auto()


class MotionPlanning(Drone):

    def __init__(self, connection):
        super().__init__(connection)

        self.target_position = np.array([0.0, 0.0, 0.0])
        self.home_position = (0.0, 0.0, 0.0)
        self.waypoints = []
        self.in_mission = True
        self.check_state = {}

        # initial state
        self.flight_state = States.MANUAL

        # register all your callbacks here
        self.register_callback(MsgID.LOCAL_POSITION, self.local_position_callback)
        self.register_callback(MsgID.LOCAL_VELOCITY, self.velocity_callback)
        self.register_callback(MsgID.STATE, self.state_callback)

    def local_position_callback(self):
        if self.flight_state == States.TAKEOFF:
            if -1.0 * self.local_position[2] > 0.95 * self.target_position[2]:
                self.waypoint_transition()
        elif self.flight_state == States.WAYPOINT:
            if np.linalg.norm(self.target_position[0:2] - self.local_position[0:2]) < 1.0:
                if len(self.waypoints) > 0:
                    self.waypoint_transition()
                else:
                    if np.linalg.norm(self.local_velocity[0:2]) < 1.0:
                        self.landing_transition()

    def velocity_callback(self):
        if self.flight_state == States.LANDING:
            if self.global_position[2] - self.global_home[2] < 0.1:
                if abs(self.local_position[2]) < 0.01:
                    self.disarming_transition()

    def state_callback(self):
        if self.in_mission:
            if self.flight_state == States.MANUAL:
                self.arming_transition()
            elif self.flight_state == States.ARMING:
                if self.armed:
                    self.plan_path()
            elif self.flight_state == States.PLANNING:
                self.takeoff_transition()
            elif self.flight_state == States.DISARMING:
                if ~self.armed & ~self.guided:
                    self.manual_transition()

    def arming_transition(self):
        self.flight_state = States.ARMING
        print("arming transition")
        self.arm()
        self.take_control()

    def takeoff_transition(self):
        self.flight_state = States.TAKEOFF
        print("takeoff transition")
        self.takeoff(self.target_position[2])

    def waypoint_transition(self):
        self.flight_state = States.WAYPOINT
        print("waypoint transition")
        self.target_position = self.waypoints.pop(0)
        print('target position', self.target_position)
        self.cmd_position(self.target_position[0], self.target_position[1], self.target_position[2], self.target_position[3])

    def landing_transition(self):
        self.flight_state = States.LANDING
        print("landing transition")
        self.land()

    def disarming_transition(self):
        self.flight_state = States.DISARMING
        print("disarm transition")
        self.disarm()
        self.release_control()

    def manual_transition(self):
        self.flight_state = States.MANUAL
        print("manual transition")
        self.stop()
        self.in_mission = False

    def send_waypoints(self):
        print("Sending waypoints to simulator ...")
        data = msgpack.dumps(self.waypoints)
        self.connection._master.write(data)

    def plan_path(self):
        self.flight_state = States.PLANNING
        print("Searching for a path ...")
        TARGET_ALTITUDE = 5
        SAFETY_DISTANCE = 5

        self.target_position[2] = TARGET_ALTITUDE

        # TODO: read lat0, lon0 from colliders into floating point values
        with open('colliders.csv') as csvfile:
            data = list(csv.reader(csvfile))
        lat0 = data[0][0].lstrip().rstrip().split(' ')[1]
        lon0 = data[0][1].lstrip().rstrip().split(' ')[1]

        # TODO: set home position to (lon0, lat0, 0)
        self.set_home_position(np.float64(lon0), np.float64(lat0), 0.)  # Home Position

        # TODO: retrieve current global position
        global_position = [self._longitude, self._latitude, self._altitude]  # Home Position => Global Position

        # TODO: convert to current local position using global_to_local()
        start_local_position = global_to_local(global_position,
                                               self.global_home)  # Global Position => Current Local Position

        # Read in obstacle map
        data = np.loadtxt('colliders.csv', delimiter=',', dtype='Float64', skiprows=2)

        # Define a grid for a particular altitude and safety margin around obstacles
        self.grid, north_offset, north_offset_max, east_offset, east_offset_max = pu.create_grid(data, TARGET_ALTITUDE, SAFETY_DISTANCE)
        '''
        self.grid, self.edges, north_offset, north_offset_max, east_offset, east_offset_max = pu.create_grid_and_edges(data,
                                                                                                             TARGET_ALTITUDE,
                                                                                                             SAFETY_DISTANCE)
        '''
        print("North offset = {0}, North max = {1}, east offset = {2}, east max = {3}".format(north_offset,
                                                                                              north_offset_max,
                                                                                              east_offset,
                                                                                              east_offset_max))

        skeleton = medial_axis(invert(self.grid))

        # Define starting point on the grid (this is just grid center)
        #  TODO: convert start position to current position rather than map center
        self.start_grid = (int(start_local_position[0] - north_offset), int(start_local_position[1] - east_offset))

        # Set goal as some arbitrary position on the grid
        goal_global_position = [-122.397347, 37.794966,
                                -0.147]  # Three Embarcadero Center, San Francisco, CA 94111, USA
        #goal_global_position = [-122.395093, 37.792088,
        #                        -0.147]  # 59 Main St, San Francisco, CA 94105, USA


        # TODO: adapt to set goal as latitude / longitude position and convert
        goal_local_position = global_to_local(goal_global_position, self.global_home)
        self.goal_grid = (int(goal_local_position[0] - north_offset), int(goal_local_position[1] - east_offset))

        print('global home: {0}'.format(self.global_home))
        print('start : global position: {0}, local position: {1}, grid: {2}'.format(self.global_position,
                                                                                    self.local_position, self.start_grid))
        print('goal : global_position = {0}, local_position: {1}, grid: {2}'.format(goal_global_position,
                                                                                    goal_local_position, self.goal_grid))


        pu.plot_graph_skeleton(self.grid, skeleton, self.start_grid, self.goal_grid)
        '''
        G = nx.Graph()
        for e in self.edges:
            p1 = e[0]
            p2 = e[1]
            dist = LA.norm(np.array(p2) - np.array(p1))
            G.add_edge(p1, p2, weight=dist)

        start_ne_g = pu.closest_point(G, self.start_grid)
        goal_ne_g = pu.closest_point(G, self.goal_grid)
        '''
        # Run A* to find a path from start to goal
        # TODO: add diagonal motions with a cost of sqrt(2) to your A* implementation
        # or move to a different search space such as a graph (not done here)
        #self.path, _ = pu.a_star(self.grid, pu.heuristic, self.start_grid, self.goal_grid)
        self.path, cost = pu.a_star(self.grid, pu.heuristic, self.start_grid, self.goal_grid)

        print('Cost = {0}'.format(cost))

        self.path = pu.collinearity_prune(self.path)

        #pu.plot_graph_a_star(self.grid, self.edges, self.path, self.start_grid, start_ne_g, self.goal_grid, goal_ne_g)
        # Convert path to waypoints
        waypoints = [[p[0] + north_offset, p[1] + east_offset, TARGET_ALTITUDE, 0] for p in self.path]
        # Set self.waypoints
        print(waypoints)
        self.waypoints = waypoints
        # TODO: send waypoints to sim (this is just for visualization of waypoints)
        self.send_waypoints()

    def start(self):
        self.start_log("Logs", "NavLog.txt")

        print("starting connection")
        self.connection.start()

        # Only required if they do threaded
        # while self.in_mission:
        #    pass

        self.stop_log()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5760, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help="host address, i.e. '127.0.0.1'")
    args = parser.parse_args()

    conn = MavlinkConnection('tcp:{0}:{1}'.format(args.host, args.port), timeout=60)
    drone = MotionPlanning(conn)
    time.sleep(1)

    #drone.make_path()
    drone.start()
    '''
    with open('colliders.csv') as csvfile:
        data = list(csv.reader(csvfile))

    print(data[0][0])
    print(data[0][1])

    lat0 = data[0][0].lstrip().rstrip().split(' ')[1]
    lon0 = data[0][1].lstrip().rstrip().split(' ')[1]

    print(lat0)
    print(lon0)
    '''
    #drone.home_position = np.array([np.float(lon0), np.float(lat0), 0.0])
    #drone.home_position = (np.float(lon0), np.float(lat0), 0.0)
    #print(drone.home_position)