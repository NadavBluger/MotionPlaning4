import math

import numpy as np
import time

import environment
from RRTTree import RRTTree

class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb, p_bias):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)
        # testing variables
        self.t_curr = 0
        self.itr_no_goal_limit = 250
        self.sample_rotation = 0.1
        # self.TWO_PI = 2 * math.pi
        self.last_cost = -1
        self.last_ratio = 1
        self.goal_prob = p_bias


    def find_path(self, start_conf, goal_conf):
        self.tree.AddVertex(start_conf)
        itrs = 0
        costs = []
        cost = math.inf
        start_time = time.time()
        while time.time() - start_time < 120:
            if self.tree.is_goal_exists(goal_conf) and self.compute_cost(self.get_path(goal_conf)) < cost:
                cost = self.compute_cost(self.get_path(goal_conf))
                costs.append((time.time() - start_time, cost))
            goal_prob = 0 if self.tree.is_goal_exists(goal_conf) else self.goal_prob
            rand_config = self.bb.sample_random_config(goal_prob, goal_conf)
            sid, nearest_config = self.tree.GetNearestVertex(rand_config)
            if self.bb.config_validity_checker(rand_config) and self.bb.edge_validity_checker(nearest_config.state,
                                                                                              rand_config):
                new_config = self.extend(nearest_config.state, rand_config)
                eid = self.tree.AddVertex(new_config)
                new_cost = self.bb.compute_distance(nearest_config.state, new_config) + self.tree.vertices[sid].cost
                self.tree.AddEdge(sid, eid, new_cost)

                nearest_neighbors, _ = self.tree.GetKNN(new_config, self.get_k())

                # Parent rewire
                potential_parents = []
                for nearest_neighbor in nearest_neighbors:
                    if self.bb.edge_validity_checker(new_config, self.tree.vertices[nearest_neighbor].state):
                        new_cost = self.bb.compute_distance(new_config, self.tree.vertices[nearest_neighbor].state) + \
                                   self.tree.vertices[nearest_neighbor].cost
                        potential_parents.append((nearest_neighbor, new_cost))
                if potential_parents:
                    potential_parents.sort(key=lambda x: x[1])
                    new_parent = potential_parents[0]
                    if new_parent[1] < self.tree.vertices[eid].cost:
                        self.tree.vertices[eid].cost = new_parent[1]
                        self.tree.edges[eid] = new_parent[0]

                # Child rewire
                potential_children = []
                for nearest_neighbor in nearest_neighbors:
                    if self.bb.edge_validity_checker(new_config, self.tree.vertices[nearest_neighbor].state):
                        new_cost = self.bb.compute_distance(new_config, self.tree.vertices[nearest_neighbor].state) + \
                                   self.tree.vertices[eid].cost
                        potential_children.append((nearest_neighbor, new_cost))
                if potential_children:
                    potential_children.sort(key=lambda x: x[1])
                    new_child = potential_children[0]
                    if new_child[1] < self.tree.vertices[eid].cost:
                        self.tree.vertices[new_child[0]].cost = new_child[1]
                        self.tree.edges[new_child] = eid
            itrs += 1
            # if self.tree.is_goal_exists(self.goal) and self.compute_cost(self.get_path()) < cost:
            #     cost = self.compute_cost(self.get_path())
            #     costs.append((itrs, cost))
            #     (costs[-1])
            if self.tree.is_goal_exists(goal_conf) and self.compute_cost(self.get_path(goal_conf)) < cost:
                cost = self.compute_cost(self.get_path(goal_conf))
                costs.append((time.time() - start_time, cost))
        print(len(self.tree.vertices))
        return self.get_path(goal_conf), costs

    def get_path(self, goal):
        if not self.tree.is_goal_exists(goal):
            return []
        current = self.tree.get_vertex_for_config(goal)
        path = [current.state]
        while self.tree.get_idx_for_config(current.state) in self.tree.edges:
            current = self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.state)]]
            path.append(current.state)
        return np.array(path)[::-1]


    def compute_cost(self, plan):
        cost = 0
        for i in range(len(plan) - 1):
            cost += self.bb.compute_distance(plan[i], plan[i + 1])
        return cost

    def extend(self, x_near, x_random):
        if self.bb.compute_distance(x_near, x_random) < self.max_step_size:
            new_conf = x_random
        else:
            new_conf = x_near + ((x_random-x_near)/self.bb.compute_distance(x_near, x_random)) * self.max_step_size
        return np.array(new_conf)

    def get_k(self):
        i = len(self.tree.vertices)
        d = len(self.tree.vertices[0].state)
        # k = e^(1+1/d)*log i
        k = int(math.exp(1 + 1 / d) * math.log(i))
        k = k if k > 1 else 1
        return min(k, len(self.tree.vertices) - 1)


class RRTMotionPlanner(object):

    def __init__(self, bb, ext_mode, goal_prob):

        # set environment and search tree
        self.bb = bb
        self.tree = RRTTree(self.bb)
        self.increment =.1

        # set search params
        self.ext_mode = ext_mode
        self.goal_prob = goal_prob

    def find_path(self, start_conf, goal_conf):
        '''
        Compute and return the plan. The function should return a numpy array containing the states in the configuration space.
        '''
        self.tree.AddVertex(start_conf)
        while not self.tree.is_goal_exists(goal_conf):
            rand_config = self.bb.sample_random_config(self.goal_prob, goal=goal_conf)
            self.extend(self.tree.GetNearestVertex(rand_config)[1], rand_config)

        current = self.tree.get_vertex_for_config(goal_conf)
        plan = [current.state]
        while np.any(current.state != start_conf):
            current = self.tree.vertices[self.tree.edges[self.tree.get_idx_for_config(current.state)]]
            plan.append(current.state)

        return np.array(plan)[::-1]

    def compute_cost(self, plan):
        '''
        Compute and return the plan cost, which is the sum of the distances between steps in the configuration space.
        @param plan A given plan for the robot.
        '''
        # The cost of the plan is the cost of the goal vertex in the tree
        cost=0
        for i in range(len(plan)-1):
            cost += self.bb.compute_distance(plan[i], plan[i+1])
        return cost


    def extend(self, near_config: np.ndarray, rand_config: np.ndarray):
        '''
        Compute and return a new configuration for the sampled one.
        @param near_config The nearest configuration to the sampled configuration.
        @param rand_config The sampled configuration.
        '''
        #E1
        if self.ext_mode == "E1":
            if not self.bb.config_validity_checker(rand_config) or not self.bb.edge_validity_checker(
                    near_config, rand_config):
                return
            eid = self.tree.AddVertex(rand_config)
            sid = self.tree.get_idx_for_config(near_config)
            self.tree.AddEdge(sid, eid, self.bb.compute_distance(near_config, rand_config))
        else:
            if self.bb.compute_distance(near_config, rand_config)<self.increment:
                new_config = rand_config
            else:
                new_config = near_config + ((rand_config - near_config) / self.bb.compute_distance(rand_config, near_config))*self.increment
            if not self.bb.config_validity_checker(new_config) or not self.bb.edge_validity_checker(
                    near_config, new_config):
                return

            eid = self.tree.AddVertex(new_config)
            sid = self.tree.get_idx_for_config(near_config)
            self.tree.AddEdge(sid, eid, self.bb.compute_distance(new_config, near_config))