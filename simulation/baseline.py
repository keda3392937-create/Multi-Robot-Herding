import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.spatial import ConvexHull
import random
import time
import datetime


class HerdingSimulation:
    def __init__(self, random_seed=None):
        if random_seed is None:
            random_seed = int(datetime.datetime.now().timestamp() * 1_000_000) % 2**32
        print(f"使用随机种子: {random_seed}")
        np.random.seed(random_seed)
        random.seed(random_seed)

        # -------------------- 参数 --------------------
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

        # -------------------- 初始化 --------------------
        self.setup_fixed_cage()
        self.initialize_positions()
        self.setup_visualization()

    # ==================== 初始化 ====================
    def setup_fixed_cage(self):
        cageWidth = 16
        cageHeight = 16
        # 将笼子固定在场地左上角；如果需要留出边距，调大 cageMargin 即可。
        cageMargin = 0
        self.cageLeft = cageMargin
        self.cageRight = self.cageLeft + cageWidth
        self.cageBottom = self.arenaSize - cageMargin - cageHeight
        self.cageTop = self.cageBottom + cageHeight
        self.cageCenter = np.array([self.cageLeft + cageWidth / 2,
                                    self.cageBottom + cageHeight / 2])

        openingSize = 6
        margin = (cageWidth - openingSize) / 2
        self.topOpeningSize = openingSize
        self.rightOpeningSize = openingSize
        self.botOpeningSize = openingSize
        self.leftOpeningSize = openingSize

        self.topOpenStart = self.cageLeft + margin
        self.topOpenEnd = self.topOpenStart + openingSize
        self.rightOpenStart = self.cageBottom + margin
        self.rightOpenEnd = self.rightOpenStart + openingSize
        self.botOpenStart = self.cageLeft + margin
        self.botOpenEnd = self.botOpenStart + openingSize
        self.leftOpenStart = self.cageBottom + margin
        self.leftOpenEnd = self.leftOpenStart + openingSize

    def initialize_positions(self):
        self.positionsEvader = np.random.rand(self.numEvaders, 2) * self.arenaSize
        self.velocitiesEvader = np.random.randn(self.numEvaders, 2)
        self.positionsHerder = np.random.rand(self.numHerders, 2) * self.arenaSize
        self.velocitiesHerder = np.zeros((self.numHerders, 2))
        self.herder_directions = np.random.rand(self.numHerders) * 2 * np.pi
    
    def initialize_herder_groups(self):
        """随机初始化herder分组"""
        # 创建herder索引列表
        herder_indices = list(range(self.numHerders))
        # 打乱顺序
        random.shuffle(herder_indices)
        
        # 初始化分组列表，每个herder所属的组
        self.herderGroups = [0] * self.numHerders
        
        # 均匀分配herder到各个组
        herders_per_group = self.numHerders // self.numGroups
        
        for group_id in range(self.numGroups):
            start_idx = group_id * herders_per_group
            end_idx = (group_id + 1) * herders_per_group
            for i in range(start_idx, end_idx):
                if i < len(herder_indices):
                    self.herderGroups[herder_indices[i]] = group_id
        
        # 打印分组信息
        print(f"Herder分组情况: {self.herderGroups}")

    # ==================== 可视化 ====================
    def setup_visualization(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.ax.set_xlim(0, self.arenaSize)
        self.ax.set_ylim(0, self.arenaSize)
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.ax.add_patch(plt.Rectangle((0, 0), self.arenaSize, self.arenaSize,
                                        fill=False, edgecolor='k', linewidth=2))
        self.draw_cage()

        self.evader_scatter = self.ax.scatter(self.positionsEvader[:, 0],
                                              self.positionsEvader[:, 1],
                                              s=30, c='yellow', marker='o', edgecolors='k')
        self.herder_scatter = self.ax.scatter(self.positionsHerder[:, 0],
                                              self.positionsHerder[:, 1],
                                              s=80, c='blue', marker='o')

        self.convex_hull_line, = self.ax.plot([], [], 'm-', linewidth=2.0)

        self.herder_fov_patches = []
        self.herder_repulsion_circles = []
        self.herder_interaction_circles = []

        for i in range(self.numHerders):
            fov_patch = self.create_fov_patch(i)
            self.herder_fov_patches.append(fov_patch)
            self.ax.add_patch(fov_patch)

            c1 = patches.Circle(self.positionsHerder[i], self.Re,
                                fill=True, facecolor='gray', alpha=0.3,
                                edgecolor='gray', linewidth=0.5)
            self.herder_repulsion_circles.append(c1)
            self.ax.add_patch(c1)

            c2 = patches.Circle(self.positionsHerder[i], self.herder_repulsion_dist,
                                fill=False, edgecolor='red', linestyle='--',
                                linewidth=1.0, alpha=0.5)
            self.herder_interaction_circles.append(c2)
            self.ax.add_patch(c2)

    def draw_cage(self):
        self.ax.plot([self.cageLeft, self.cageLeft],
                     [self.cageBottom, self.leftOpenStart], 'g-', linewidth=2)
        self.ax.plot([self.cageLeft, self.cageLeft],
                     [self.leftOpenEnd, self.cageTop], 'g-', linewidth=2)
        self.ax.plot([self.cageLeft, self.topOpenStart],
                     [self.cageTop, self.cageTop], 'g-', linewidth=2)
        self.ax.plot([self.topOpenEnd, self.cageRight],
                     [self.cageTop, self.cageTop], 'g-', linewidth=2)
        self.ax.plot([self.cageRight, self.cageRight],
                     [self.cageTop, self.rightOpenEnd], 'g-', linewidth=2)
        self.ax.plot([self.cageRight, self.cageRight],
                     [self.rightOpenStart, self.cageBottom], 'g-', linewidth=2)
        self.ax.plot([self.cageRight, self.botOpenEnd],
                     [self.cageBottom, self.cageBottom], 'g-', linewidth=2)
        self.ax.plot([self.botOpenStart, self.cageLeft],
                     [self.cageBottom, self.cageBottom], 'g-', linewidth=2)

    def create_fov_patch(self, herder_idx):
        herder_pos = self.positionsHerder[herder_idx]
        herder_dir = self.herder_directions[herder_idx]
        start_angle = np.degrees(herder_dir - self.fov_rad / 2)
        end_angle = np.degrees(herder_dir + self.fov_rad / 2)
        return patches.Wedge(herder_pos, self.Rh, start_angle, end_angle,
                             fill=False, edgecolor=[0.7, 0.7, 1],
                             linestyle='--', linewidth=1)

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

    # ==================== Evader 动力学 ====================
    def update_evaders(self):
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

            self.velocitiesEvader[e] += accelerationEvader * self.dt
            speed = np.linalg.norm(self.velocitiesEvader[e])
            if speed > self.speedLimitEvader:
                self.velocitiesEvader[e] = self.velocitiesEvader[e] / speed * self.speedLimitEvader

            newPos = self.positionsEvader[e] + self.velocitiesEvader[e] * self.dt
            newPos = np.clip(newPos, 0, self.arenaSize)
            self.positionsEvader[e] = newPos

    def calculate_cage_force(self, e):
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
        elif x > self.arenaSize - self.edgeRepulsionDist:
            boundaryForce[0] = -self.edgeRepulsionCoeff / ((self.arenaSize - x) ** 2 + 1e-10)
        if y < self.edgeRepulsionDist:
            boundaryForce[1] = self.edgeRepulsionCoeff / (y ** 2 + 1e-10)
        elif y > self.arenaSize - self.edgeRepulsionDist:
            boundaryForce[1] = -self.edgeRepulsionCoeff / ((self.arenaSize - y) ** 2 + 1e-10)
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

    # ==================== Herder 动力学 ====================
    def update_herders(self, t):
        accelerationsHerder = np.zeros((self.numHerders, 2))

        if t >= self.waitTime:
            accelerationsHerder = self.update_herders_with_convex_hull(t)

        self.velocitiesHerder += accelerationsHerder * self.dt
        for herder_idx in range(self.numHerders):
            speed = np.linalg.norm(self.velocitiesHerder[herder_idx])
            if speed > self.baseSpeedLimitHerder:
                self.velocitiesHerder[herder_idx] = self.velocitiesHerder[herder_idx] / speed * self.baseSpeedLimitHerder

            newPos = self.positionsHerder[herder_idx] + self.velocitiesHerder[herder_idx] * self.dt
            x, y = newPos
            if (self.cageLeft <= x <= self.cageRight and
                    self.cageBottom <= y <= self.cageTop):
                to_center = self.cageCenter - self.positionsHerder[herder_idx]
                dist = np.linalg.norm(to_center)
                repel_dir = -to_center / dist if dist > 0 else np.array([1, 1])
                self.velocitiesHerder[herder_idx] = self.baseSpeedLimitHerder * repel_dir
                newPos = self.positionsHerder[herder_idx] + self.velocitiesHerder[herder_idx] * self.dt

            newPos = np.clip(newPos, 0, self.arenaSize)
            self.positionsHerder[herder_idx] = newPos

    # -------------------- 辅助 --------------------
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

    def update_convex_hull_visualization(self):
        if self.numHerders < 1:
            self.convex_hull_line.set_data([], [])
            return

        try:
            _, hull_points = self.get_augmented_hull_entities(list(range(self.numHerders)))
            hull_points_closed = np.vstack([hull_points, hull_points[0]])
            self.convex_hull_line.set_data(hull_points_closed[:, 0], hull_points_closed[:, 1])
        except:
            self.convex_hull_line.set_data([], [])

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

            convex_hull_points_closed = np.vstack([convex_hull_points, convex_hull_points[0]])
            self.convex_hull_line.set_data(convex_hull_points_closed[:, 0], convex_hull_points_closed[:, 1])
        except:
            hull_entities = [{
                "type": "herder",
                "index": group_idx,
                "position": self.positionsHerder[group_idx],
            } for group_idx in group_indices]
            convex_hull_indices_global = list(group_indices)
            self.convex_hull_line.set_data([], [])

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

    # ==================== 可视化更新 ====================
    def update_visualization(self, t):
        self.evader_scatter.set_offsets(self.positionsEvader)
        self.herder_scatter.set_offsets(self.positionsHerder)

        self.update_convex_hull_visualization()

        for i in range(self.numHerders):
            self.herder_fov_patches[i].remove()
            self.herder_fov_patches[i] = self.create_fov_patch(i)
            self.ax.add_patch(self.herder_fov_patches[i])
            self.herder_repulsion_circles[i].center = tuple(self.positionsHerder[i])
            self.herder_interaction_circles[i].center = tuple(self.positionsHerder[i])

        # 仅保留极简标题
        isInCage = ((self.positionsEvader[:, 0] >= self.cageLeft) &
                    (self.positionsEvader[:, 0] <= self.cageRight) &
                    (self.positionsEvader[:, 1] >= self.cageBottom) &
                    (self.positionsEvader[:, 1] <= self.cageTop))
        inCageCount = np.sum(isInCage)
        self.ax.set_title(f'Time = {t:.1f}s | Evaders in cage: {inCageCount}/{self.numEvaders}')
        plt.draw()
        plt.pause(0.01)
        return np.all(isInCage)

    # ==================== 主循环 ====================
    def run_simulation(self):
        print("开始仿真（多 herder 模式）...")
        print(f"笼子位置: ({self.cageLeft:.1f}, {self.cageBottom:.1f}) - "
              f"({self.cageRight:.1f}, {self.cageTop:.1f})")
        print(f"开口大小: 上{self.topOpeningSize:.1f} 右{self.rightOpeningSize:.1f} "
              f"下{self.botOpeningSize:.1f} 左{self.leftOpeningSize:.1f}")
        print(f"Herder视野: {self.fov_angle}° 扇形")
        print(f"Herder总数: {self.numHerders}个")
        print("模式: 全程多 herder 模式")

        start_wall = time.time()
        for t in np.arange(0, self.totalTime, self.dt):
            self.update_evaders()
            self.update_herders(t)
            all_in = self.update_visualization(t)
            if all_in:
                print(f'放牧完成 {t:.1f}s!')
                break

        end_wall = time.time()
        print("\n=== 仿真统计 ===")
        print(f"总运行时间: {end_wall - start_wall:.2f}秒")
        isInCage = ((self.positionsEvader[:, 0] >= self.cageLeft) &
                    (self.positionsEvader[:, 0] <= self.cageRight) &
                    (self.positionsEvader[:, 1] >= self.cageBottom) &
                    (self.positionsEvader[:, 1] <= self.cageTop))
        final_count = np.sum(isInCage)
        print(f"最终笼子内evader数量: {final_count}/{self.numEvaders} "
              f"({final_count / self.numEvaders * 100:.1f}%)")
        plt.show()


# ==================== 运行 ====================
if __name__ == "__main__":
    sim = HerdingSimulation()
    sim.run_simulation()
