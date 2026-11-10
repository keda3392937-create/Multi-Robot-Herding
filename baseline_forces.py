"""Headless force equations ported from simulation/baseline.py.

The convex-hull, FOV, density and pursuit expressions are retained from the
uploaded simulation. Real positions are supplied externally; no simulated
position integration, np.clip, matplotlib, or random initial positions here.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


class BaselineForces:
    def __init__(self):
        self.numEvaders = 30
        self.numHerders = 8
        self.arenaSize = 80
        self.Re = 6
        self.Rh = 12
        self.speedLimitEvader = 3  # 减小evader速度限制
        self.baseSpeedLimitHerder = 2  # 减小herder速度限制
        self.dt = 0.1
        self.totalTime = 1500  # 总时限1500秒
        self.edgeRepulsionDist = 5  # 增大排斥距离，使减速更平滑
        self.edgeRepulsionCoeff = 50  # 减小排斥系数，使减速更柔和

        self.dispersionCoeff = 150  # 减小分散系数，减小加速度
        self.aggregationCoeff = 1  # 减小聚集系数，减小加速度

        self.herder_repulsion_dist = 8.0
        self.herder_repulsion_coeff = 100  # 减小herder排斥系数
        self.min_herder_distance = 6.0

        self.lambda_val = self.baseSpeedLimitHerder / self.speedLimitEvader
        self.theta_i = 2 * np.arcsin(self.lambda_val)
        self.h_i = 0.1  # 减小herder加速度系数
        self.k_i = 0.5
        self.dc = 2
        self.waitTime = 1

        self.fov_angle = 120
        self.fov_rad = np.radians(self.fov_angle)
        self.rotation_speed = 0.5

        self.densityThreshold = 3
        self.maxDensity = 20
        self.minSpeedFactor = 0.3
        self.minSpeedFactorNearCage = 0.1

        self.arenaWidth = self.arenaSize
        self.arenaHeight = self.arenaSize
        self.virtual_target = True

    def get_cage_hull_vertices(self):
        return np.array([
            [self.cageLeft, self.cageTop],
            [self.cageLeft, self.cageBottom],
            [self.cageRight, self.cageTop],
        ])

    def get_augmented_hull_entities(self, group_indices):
        group_positions = self.positionsHerder[group_indices]
        cage_vertices = self.get_cage_hull_vertices()
        augmented_points = np.vstack([group_positions, cage_vertices])

        hull = ConvexHull(augmented_points)
        hull_vertex_indices = hull.vertices
        hull_points = augmented_points[hull_vertex_indices]

        hull_entities = []
        for vertex_idx in hull_vertex_indices:
            if vertex_idx < len(group_indices):
                hull_entities.append({
                    "type": "herder",
                    "index": group_indices[vertex_idx],
                    "position": augmented_points[vertex_idx],
                })
            else:
                hull_entities.append({
                    "type": "cage_vertex",
                    "index": vertex_idx - len(group_indices),
                    "position": augmented_points[vertex_idx],
                })

        return hull_entities, hull_points

    def evader_accelerations(self):
        accelerations = np.zeros((self.numEvaders, 2))
        isInCage = ((self.positionsEvader[:, 0] >= self.cageLeft) &
                    (self.positionsEvader[:, 0] <= self.cageRight) &
                    (self.positionsEvader[:, 1] >= self.cageBottom) &
                    (self.positionsEvader[:, 1] <= self.cageTop))

        for e in range(self.numEvaders):
            dist_vec_all = self.positionsEvader - self.positionsEvader[e]
            distances = np.linalg.norm(dist_vec_all, axis=1)
            neighbors = np.where((distances < self.Re) & (distances > 0))[0]

            if len(neighbors) > 0:
                deltaPositions = self.positionsEvader[e] - self.positionsEvader[neighbors]
                dispersionForce = np.sum(deltaPositions /
                                         (distances[neighbors][:, np.newaxis] ** 3 + 1e-10),
                                         axis=0) / len(neighbors)
            else:
                dispersionForce = np.zeros(2)

            aggregationForce = np.zeros(2)
            isHerderDetected = False
            for j in range(self.numHerders):
                if np.linalg.norm(self.positionsEvader[e] - self.positionsHerder[j]) < self.Re:
                    isHerderDetected = True
                    if len(neighbors) > 0:
                        aggregationForce += np.mean(self.positionsEvader[neighbors] -
                                                    self.positionsEvader[e], axis=0)

            escapeForce = np.zeros(2)
            for j in range(self.numHerders):
                dist = np.linalg.norm(self.positionsEvader[e] - self.positionsHerder[j])
                if dist < self.Re:
                    escapeForce += (self.positionsEvader[e] - self.positionsHerder[j]) / (dist ** 3 + 1e-10)

            cageForce = self.calculate_cage_force(e)
            boundaryForce = self.calculate_boundary_force(e)
            cageCenterForce = self.calculate_cage_center_force(e, isInCage)

            if isInCage[e]:
                accelerationEvader = (0.01 * self.dispersionCoeff * dispersionForce +
                                      10 * cageCenterForce)
            else:
                if isHerderDetected:
                    accelerationEvader = (1e6 * escapeForce + cageForce + boundaryForce +
                                          self.dispersionCoeff * dispersionForce +
                                          self.aggregationCoeff * aggregationForce)
                else:
                    accelerationEvader = (self.dispersionCoeff * dispersionForce +
                                          cageForce + boundaryForce)

            accelerations[e] = accelerationEvader
        return accelerations

    def calculate_cage_force(self, e):
        # A target region is traversable. The simulation's drawn openings
        # were not respected by this wall force; disable it for virtual cages.
        if self.virtual_target:
            return np.zeros(2)
        x, y = self.positionsEvader[e]
        cageForce = np.zeros(2)
        # 顶部边缘斥力（向上）：将笼外evader推离笼子顶部
        if self.cageLeft <= x <= self.cageRight and self.cageTop <= y <= self.cageTop + self.edgeRepulsionDist:
            cageForce[1] = self.edgeRepulsionCoeff / ((y - self.cageTop) ** 2 + 1e-10)
        # 底部边缘斥力（向下）：将笼外evader推离笼子底部
        elif self.cageLeft <= x <= self.cageRight and self.cageBottom - self.edgeRepulsionDist <= y <= self.cageBottom:
            cageForce[1] = -self.edgeRepulsionCoeff / ((y - self.cageBottom) ** 2 + 1e-10)
        # 右侧边缘斥力（向右）：将笼外evader推离笼子右侧
        elif self.cageBottom <= y <= self.cageTop and self.cageRight <= x <= self.cageRight + self.edgeRepulsionDist:
            cageForce[0] = self.edgeRepulsionCoeff / ((x - self.cageRight) ** 2 + 1e-10)
        # 左侧边缘斥力（向左）：将笼外evader推离笼子左侧
        elif self.cageBottom <= y <= self.cageTop and self.cageLeft - self.edgeRepulsionDist <= x <= self.cageLeft:
            cageForce[0] = -self.edgeRepulsionCoeff / ((x - self.cageLeft) ** 2 + 1e-10)
        return cageForce

    def calculate_boundary_force(self, e):
        x, y = self.positionsEvader[e]
        boundaryForce = np.zeros(2)
        if x < self.edgeRepulsionDist:
            boundaryForce[0] = self.edgeRepulsionCoeff / (x ** 2 + 1e-10)
        elif x > self.arenaWidth - self.edgeRepulsionDist:
            boundaryForce[0] = -self.edgeRepulsionCoeff / ((self.arenaWidth - x) ** 2 + 1e-10)
        if y < self.edgeRepulsionDist:
            boundaryForce[1] = self.edgeRepulsionCoeff / (y ** 2 + 1e-10)
        elif y > self.arenaHeight - self.edgeRepulsionDist:
            boundaryForce[1] = -self.edgeRepulsionCoeff / ((self.arenaHeight - y) ** 2 + 1e-10)
        return boundaryForce

    def calculate_cage_center_force(self, e, isInCage):
        x, y = self.positionsEvader[e]
        cageCenterForce = np.zeros(2)
        if isInCage[e]:
            if x < self.cageLeft + self.edgeRepulsionDist:
                cageCenterForce[0] = self.edgeRepulsionCoeff / ((x - self.cageLeft) ** 2 + 1e-10)
            elif x > self.cageRight - self.edgeRepulsionDist:
                cageCenterForce[0] = -self.edgeRepulsionCoeff / ((self.cageRight - x) ** 2 + 1e-10)
            if y < self.cageBottom + self.edgeRepulsionDist:
                cageCenterForce[1] = self.edgeRepulsionCoeff / ((y - self.cageBottom) ** 2 + 1e-10)
            elif y > self.cageTop - self.edgeRepulsionDist:
                cageCenterForce[1] = -self.edgeRepulsionCoeff / ((self.cageTop - y) ** 2 + 1e-10)
        return cageCenterForce

    def get_visible_evaders(self, herder_idx):
        herder_pos = self.positionsHerder[herder_idx]
        herder_dir = self.herder_directions[herder_idx]
        visible = []
        for e in range(self.numEvaders):
            if self.is_in_cage(self.positionsEvader[e]):
                continue
            rel = self.positionsEvader[e] - herder_pos
            dist = np.linalg.norm(rel)
            if dist > self.Rh or dist == 0:
                continue
            angle = np.arctan2(rel[1], rel[0])
            angle_diff = (angle - herder_dir + np.pi) % (2 * np.pi) - np.pi
            if abs(angle_diff) <= self.fov_rad / 2:
                visible.append(e)
        return np.array(visible, dtype=int)

    def is_in_cage(self, pos):
        x, y = pos
        return (self.cageLeft <= x <= self.cageRight and
                self.cageBottom <= y <= self.cageTop)

    def calculate_speed_limit(self, herder_idx):
        visible = self.get_visible_evaders(herder_idx)
        sector_area = 0.5 * self.Rh ** 2 * self.fov_rad
        density = len(visible) / sector_area if sector_area > 0 else 0
        speed = (self.baseSpeedLimitHerder *
                 max(self.minSpeedFactor,
                     1 - (density - self.densityThreshold) /
                     (self.maxDensity - self.densityThreshold)))
        return max(self.baseSpeedLimitHerder * self.minSpeedFactor,
                   min(self.baseSpeedLimitHerder, speed))

    def update_herder_direction(self, herder_idx):
        visible = self.get_visible_evaders(herder_idx)
        if len(visible) > 0:
            closest = visible[np.argmin(
                [np.linalg.norm(self.positionsEvader[e] - self.positionsHerder[herder_idx])
                 for e in visible])]
            rel = self.positionsEvader[closest] - self.positionsHerder[herder_idx]
            self.herder_directions[herder_idx] = np.arctan2(rel[1], rel[0])
        else:
            self.herder_directions[herder_idx] += self.rotation_speed * self.dt
            self.herder_directions[herder_idx] %= (2 * np.pi)

    def update_herders_with_convex_hull(self, t):
        accelerationsHerder = np.zeros((self.numHerders, 2))

        if t >= self.waitTime and self.numHerders >= 1:
            group_indices = list(range(self.numHerders))
            accelerationsHerder = self.update_herders_group(group_indices, t)

        return accelerationsHerder

    def update_herders_group(self, group_indices, t):
        if len(group_indices) < 1:
            return np.zeros((self.numHerders, 2))

        accelerationsHerder = np.zeros((self.numHerders, 2))

        try:
            hull_entities, convex_hull_points = self.get_augmented_hull_entities(group_indices)
            convex_hull_indices_global = [entity["index"] for entity in hull_entities
                                          if entity["type"] == "herder"]

        except (QhullError, ValueError):
            hull_entities = [{
                "type": "herder",
                "index": group_idx,
                "position": self.positionsHerder[group_idx],
            } for group_idx in group_indices]
            convex_hull_indices_global = list(group_indices)

        if len(convex_hull_indices_global) >= 1:
            self.update_convex_hull_herders(hull_entities, accelerationsHerder)

        non_hull_indices = [i for i in group_indices if i not in convex_hull_indices_global]
        self.update_non_convex_hull_herders(non_hull_indices, accelerationsHerder)

        return accelerationsHerder

    def update_convex_hull_herders(self, hull_entities, accelerationsHerder):
        if not hull_entities:
            return

        hull_positions = np.array([entity["position"] for entity in hull_entities])
        reference_point = self.cageCenter

        delta = hull_positions - reference_point
        r_i = np.sqrt(np.sum(delta ** 2, axis=1))
        alpha_i = np.arctan2(delta[:, 1], delta[:, 0])
        alpha_i[alpha_i < 0] += 2 * np.pi

        hull_sorted_indices = np.argsort(alpha_i)
        sorted_hull_entities = [hull_entities[i] for i in hull_sorted_indices]
        alpha_i = alpha_i[hull_sorted_indices]
        r_i = r_i[hull_sorted_indices]

        num_hull_vertices = len(hull_entities)
        Psi = np.zeros(num_hull_vertices)
        for i in range(num_hull_vertices):
            if i < num_hull_vertices - 1:
                Psi[i] = alpha_i[i + 1] - alpha_i[i] - self.theta_i / 2
            else:
                Psi[i] = alpha_i[0] - alpha_i[-1] + 2 * np.pi - self.theta_i / 2

        for idx, entity in enumerate(sorted_hull_entities):
            if entity["type"] != "herder":
                continue
            self.update_single_convex_herder(entity["index"], idx, Psi, r_i, alpha_i,
                                             num_hull_vertices, accelerationsHerder, reference_point)

    def update_non_convex_hull_herders(self, non_hull_indices, accelerationsHerder):
        reference_point = self.cageCenter

        for i in non_hull_indices:
            currentHerderPos = self.positionsHerder[i]
            speed_limit = self.calculate_speed_limit(i)

            # 获取当前herder视野内的笼子外evader
            visible_evaders = self.get_visible_evaders(i)
            target = reference_point
            
            # 如果视野内有笼子外evader，选择最近的作为目标
            if len(visible_evaders) > 0:
                closest_e = visible_evaders[np.argmin(
                    [np.linalg.norm(self.positionsEvader[e] - currentHerderPos)
                     for e in visible_evaders])]
                target = self.positionsEvader[closest_e]
            # 如果视野内没有evader，使用离笼子最远的evader
            else:
                outside_idx = [e for e in range(self.numEvaders)
                              if not self.is_in_cage(self.positionsEvader[e])]
                if len(outside_idx) > 0:
                    farthest_e = max(outside_idx,
                                  key=lambda e: np.linalg.norm(self.positionsEvader[e] - self.cageCenter))
                    target = self.positionsEvader[farthest_e]

            dir_hunt = target - currentHerderPos
            dir_hunt = dir_hunt / (np.linalg.norm(dir_hunt) + 1e-10)

            totalForce = self.h_i * np.linalg.norm(currentHerderPos - reference_point) * dir_hunt

            for other_herder_idx in range(self.numHerders):
                if other_herder_idx == i:
                    continue
                other_pos = self.positionsHerder[other_herder_idx]
                dist_vec = currentHerderPos - other_pos
                dist = np.linalg.norm(dist_vec)
                if dist < self.herder_repulsion_dist and dist > 0:
                    repulsion_force = (self.herder_repulsion_coeff / (dist ** 2 + 1e-10)) * (dist_vec / dist)
                    totalForce += repulsion_force

            force_norm = np.linalg.norm(totalForce)
            if force_norm > speed_limit:
                totalForce = totalForce / force_norm * speed_limit

            accelerationsHerder[i] = totalForce
            self.update_herder_direction(i)

    def update_single_convex_herder(self, herder_idx, idx, Psi, r_i, alpha_i,
                                    num_hull_vertices, accelerationsHerder, reference_point):
        currentHerderPos = self.positionsHerder[herder_idx]
        speedLimitHerder = self.calculate_speed_limit(herder_idx)

        if idx == 0:
            Psi_diff = Psi[0] - Psi[-1]
        else:
            Psi_diff = Psi[idx] - Psi[idx - 1]

        delta_i = 2 * abs(Psi_diff) / (4 * np.pi - self.theta_i)
        sum_r = (r_i[idx] + r_i[(idx - 2) % num_hull_vertices] +
                 r_i[idx % num_hull_vertices])
        gamma_i = np.sin(np.pi * (r_i[idx] / sum_r)) * np.log2(3)
        beta_i = np.pi / 2 * (1 - np.exp(-delta_i * gamma_i))

        dir_surround = np.array([-np.sin(alpha_i[idx]), np.cos(alpha_i[idx])])

        # 获取当前herder视野内的笼子外evader
        visible_evaders = self.get_visible_evaders(herder_idx)
        target = reference_point
        
        # 如果视野内有笼子外evader，选择最近的作为目标
        if len(visible_evaders) > 0:
            closest_e = visible_evaders[np.argmin(
                [np.linalg.norm(self.positionsEvader[e] - currentHerderPos)
                 for e in visible_evaders])]
            target = self.positionsEvader[closest_e]
        # 如果视野内没有evader，使用离笼子最远的evader
        else:
            outside_idx = [e for e in range(self.numEvaders)
                          if not self.is_in_cage(self.positionsEvader[e])]
            if len(outside_idx) > 0:
                farthest_e = max(outside_idx,
                              key=lambda e: np.linalg.norm(self.positionsEvader[e] - self.cageCenter))
                target = self.positionsEvader[farthest_e]

        dir_hunt = target - currentHerderPos
        dir_hunt = dir_hunt / (np.linalg.norm(dir_hunt) + 1e-10)

        v_is = self.k_i * r_i[idx] * Psi_diff * dir_surround * np.sin(beta_i)
        v_ih = self.h_i * r_i[idx] * dir_hunt * np.cos(beta_i)
        totalForce = v_is + v_ih

        for other_herder_idx in range(self.numHerders):
            if other_herder_idx == herder_idx:
                continue
            other_pos = self.positionsHerder[other_herder_idx]
            dist_vec = currentHerderPos - other_pos
            dist = np.linalg.norm(dist_vec)
            if dist < self.herder_repulsion_dist and dist > 0:
                repulsion_force = (self.herder_repulsion_coeff / (dist ** 2 + 1e-10)) * (dist_vec / dist)
                totalForce += repulsion_force

        force_norm = np.linalg.norm(totalForce)
        if force_norm > speedLimitHerder:
            totalForce = totalForce / force_norm * speedLimitHerder

        accelerationsHerder[herder_idx] = totalForce
        self.update_herder_direction(herder_idx)

