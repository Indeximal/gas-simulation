from collections import deque
import csv

import numpy as np
import pygame


# Freestanding utils
def calc_normal(arr):
    return np.array(arr) / np.linalg.norm(arr)

def clamp_index(n, lenght):
    return max(min(n, lenght - 1), 0)

def is_valid_index(index, container):
    i_arr = np.array(index)
    shape = np.array(container.shape)
    return (i_arr < shape).all() and (i_arr >= 0).all()

def indices(x, y):
    return [(i, j) for i in range(x) for j in range(y)]

def neighbor_buckets(index):
    i, j = index
    return [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]


class Particle:
    def __init__(self, pos, vel, mass, radius, flag=""):
        self.pos = np.array(pos).astype(float)
        self.vel = np.array(vel).astype(float)
        self.mass = float(mass)
        self.radius = float(radius)
        self.flag = flag

    def tick(self, dt):
        self.vel += PHYSICS.gravity * dt
        self.pos += self.vel * dt

    def collide_with_surface(self, surface_normal):
        normal = calc_normal(surface_normal)
        if np.dot(self.vel, normal) < 0:
            new_vel = self.vel - 2 * np.dot(self.vel, normal) * normal
            self.vel = new_vel

    def draw(self, screen):
        r = int(self.radius)
        V = np.linalg.norm(self.vel) * .01
        col_t = (-1 / (V + 1)) + 1
        c0 = np.array([0, 0, 0]) ** 2
        c1 = np.array([252, 185, 30]) ** 2
        c = tuple(np.sqrt(c1 * col_t + c0 * (1 - col_t)))
        p = tuple(self.pos.astype(int))
        pygame.draw.circle(screen, c, p, r, 0)


class Colored_Particle(Particle):
    def __init__(self, pos, vel, mass, radius, color, flag=""):
        super(Colored_Particle, self).__init__(pos, vel, mass, radius, flag)
        self.color = color

    def draw(self, screen):
        r = int(self.radius)
        p = tuple(self.pos.astype(int))
        pygame.draw.circle(screen, self.color, p, r + 2, 0)
        super(Colored_Particle, self).draw(screen)      


class Bond:
    def __init__(self, obj1, obj2, constant, lenght):
        self.obj1 = obj1
        self.obj2 = obj2
        self.constant = constant
        self.lenght = lenght

    def tick(self, dt):
        diff = self.obj1.pos - self.obj2.pos
        distance = np.linalg.norm(diff)
        normal = (diff / distance)
        deflection = distance - self.lenght
        dv = - normal * deflection * self.constant * dt
        self.obj1.vel += dv
        self.obj2.vel -= dv

    def draw(self, screen):
        pygame.draw.line(screen, (128, 128, 128), self.obj1.pos, self.obj2.pos, 3)


def get_avg_pos_for_flag(flat_bucket_iter, flag):
    total = np.zeros(2)
    count = 0
    for bucket in flat_bucket_iter:
        for obj in bucket:
            if obj.flag == flag:
                total += obj.pos
                count += 1
    return total / count


def get_speed_histogram(objects_list, bins=50, bin_size=15):
    counts = np.zeros(bins)
    for bucket in objects_list:
        for obj in bucket:
            V = np.linalg.norm(obj.vel)
            b = int(V / bin_size)
            if b < bins:
                counts[b] += 1
    return counts


def draw_histogram(screen, hist_data, max_height=100,
                   color=(200, 200, 200)):
    scale = max_height / max(hist_data)
    bins = len(hist_data)

    width = screen.get_width() / bins
    screen_height = screen.get_height()
    for i, height in enumerate(hist_data * scale):
        rect = (i * width, screen_height, width, -height)
        pygame.draw.rect(screen, color, rect, 0)


def random_helium(energy):
    helium_mass = 4
    helium_radius = 6
    helium_color_1 = (150, 150, 255)
    helium_color_2 = (255, 50, 50)
    if PHYSICS.gravity[1] != 0:
        max_height = min(energy / abs(PHYSICS.gravity[1]) / helium_mass, screen_size[1])
    else:
        max_height = screen_size[1]
    pos = np.random.random(2) * np.array((screen_size[0], max_height))
    e_pot = abs(PHYSICS.gravity[1]) * helium_mass * pos[1]
    e_kin = energy - e_pot
    speed = np.sqrt(2 * e_kin / helium_mass)
    angle = np.random.rand() * np.pi * 2
    vel = np.array([np.cos(angle), np.sin(angle)]) * speed
    color = helium_color_1 if pos[0] < screen_size[0] / 2 else helium_color_2
    flag = "left" if pos[0] < screen_size[0] / 2 else "right"
    return Colored_Particle(pos, vel, helium_mass, helium_radius, color, flag=flag)


# TODO ENERGY
def random_hydogen2(speed):
    hydrogen_mass = 1
    hydrogen_radius = 4
    h2_bond_len = 12
    h2_bond_strength = 1000
    angle = np.random.rand() * np.pi * 2
    vel = np.array([np.cos(angle), np.sin(angle)]) * speed
    pos = np.random.random(2) * np.array(screen_size)
    pos2 = pos + [0, h2_bond_len]
    h1 = Particle(pos, vel, hydrogen_mass, hydrogen_radius)
    h2 = Particle(pos2, vel, hydrogen_mass, hydrogen_radius)
    bond = Bond(h1, h2, h2_bond_strength, h2_bond_len)
    return (h1, h2, bond)


def calc_bucket(obj, buckets_shape, screen_size):
    buckets_x, buckets_y = buckets_shape
    bucket_size_x = screen_size[0] / buckets_x
    bucket_size_y = screen_size[1] / buckets_y

    new_bucket_x = clamp_index(int(obj.pos[0] // bucket_size_x), buckets_x)
    new_bucket_y = clamp_index(int(obj.pos[1] // bucket_size_y), buckets_y)

    return (new_bucket_x, new_bucket_y)


class PHYSICS:
    gravity = np.array([0., -0.])


# Init Pygame
pygame.init()
pygame.font.init()

screen_size = width, height = 600, 200
sizeArr = np.array(screen_size)

screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()

# Init Objects
bucket_count = buckets_x, buckets_y = 8, 3
physics_buckets = np.empty(bucket_count, dtype=list)
bonds = list()
for i, j in indices(buckets_x, buckets_y):
    physics_buckets[i, j] = list()

# Generate helium objects
for obj in [random_helium(e) for e in np.ones(150) * 200_000]:
    b = calc_bucket(obj, bucket_count, screen_size)
    physics_buckets[b].append(obj)

# Generate hydrogen molecules
for o1, o2, bond in [random_hydogen2(e) for e in np.ones(0) * 500]:
    b1 = calc_bucket(o1, bucket_count, screen_size)
    physics_buckets[b1].append(o1)
    b2 = calc_bucket(o2, bucket_count, screen_size)
    physics_buckets[b2].append(o2)
    bonds.append(bond)

# Init graphics and data collection
total_speed_hist = None
debug_font = pygame.font.Font(None, 24)
checks_deque = deque(maxlen=10)
coll_deque = deque(maxlen=10)

# Main Loop
running = True
simulating = True
ticks = 0
dt = 0.01
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
        # Key event
        if event.type == pygame.KEYDOWN:
            # Pause
            if event.key == pygame.K_SPACE:
                simulating = not simulating
            # Quit
            if event.key == pygame.K_ESCAPE:
                running = False
            # Reset Histogram
            if event.key == pygame.K_r:
                total_speed_hist = None
        # Resize
        if event.type == pygame.VIDEORESIZE:
            screen_size = width, height = event.w, event.h
            screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)

    frame_time = clock.tick() / 1000.

    if not simulating:
        continue

    if ticks in (0, 250, 1000):
        simulating = False

    ticks += 1

    # Move every Particle
    for bucket in physics_buckets.flat:
        for obj in bucket:
            obj.tick(dt)

    # Update buckets
    for i, rows in enumerate(physics_buckets):
        for j, bucket in enumerate(rows):
            for obj in bucket:
                b = calc_bucket(obj, bucket_count, screen_size)
                if b != (i, j):
                    bucket.remove(obj)
                    physics_buckets[b].append(obj)

    # Particle-Wall collisions, optimized
    for bucket in physics_buckets[0]:
        for obj in bucket:
            if obj.pos[0] < obj.radius:
                obj.collide_with_surface((1, 0))
    for bucket in physics_buckets[-1]:
        for obj in bucket:
            if obj.pos[0] >= width - obj.radius:
                obj.collide_with_surface((-1, 0))
    for bucket in physics_buckets[..., 0]:
        for obj in bucket:
            if obj.pos[1] < obj.radius:
                obj.collide_with_surface((0, 1))
    for bucket in physics_buckets[..., -1]:
        for obj in bucket:
            if obj.pos[1] >= height - obj.radius:
                obj.collide_with_surface((0, -1))

    check_counter = 0
    coll_counter = 0

    # Particle-Particle collisions
    for bucket_index in indices(buckets_x, buckets_y):
        for other_index in neighbor_buckets(bucket_index):
            if not is_valid_index(other_index, physics_buckets):
                continue
            for i, obj1 in enumerate(physics_buckets[bucket_index]):
                # Prevents double calculation where obj1 and obj2 are swapped
                start_index = i + 1 if bucket_index == other_index else 0
                for obj2 in physics_buckets[other_index][start_index:]:
                    check_counter += 1
                    # Check collision with distance
                    max_dist = (obj1.radius + obj2.radius) ** 2
                    if sum((obj1.pos - obj2.pos) ** 2) > max_dist:
                        continue
                    coll_counter += 1
                    # Formula from https://en.wikipedia.org/wiki/Elastic_collision#Two-dimensional_collision_with_two_moving_objects
                    # With explanation form https://stackoverflow.com/questions/35211114/2d-elastic-ball-collision-physics
                    mass_scalar_1 = (2 * obj2.mass) / (obj1.mass + obj2.mass)
                    mass_scalar_2 = (2 * obj1.mass) / (obj1.mass + obj2.mass)
                    pos_diff = obj1.pos - obj2.pos
                    # Prevents countinuos collisions by checking whether the two
                    # objects move towards each other.
                    if np.dot(pos_diff, obj1.vel) > 0 and np.dot(pos_diff, obj2.vel) < 0:
                        continue
                    dot_scalar = np.dot(obj1.vel - obj2.vel, pos_diff) / sum(pos_diff ** 2)
                    vel_1 = obj1.vel - mass_scalar_1 * dot_scalar * pos_diff
                    vel_2 = obj2.vel + mass_scalar_1 * dot_scalar * pos_diff
                    obj1.vel = vel_1
                    obj2.vel = vel_2

    # Update covalent bonds
    for bond in bonds:
        bond.tick(dt)


    # Draw background
    screen.fill((255, 255, 255))

    # Draw Bonds
    for bond in bonds:
        bond.draw(screen)

    # Draw objects
    for bucket in physics_buckets.flat:
        for obj in bucket:
            obj.draw(screen)

    checks_deque.append(check_counter)
    coll_deque.append(coll_counter)

    # Draw text
    def render_text(text, y):
        text_surface = debug_font.render(text, True, (0, 0, 0))
        screen.blit(text_surface, (10, y))

    simulation_speed = 1 / frame_time
    render_text("tick {} @ {:.3f} t/s".format(ticks, simulation_speed), 10)
    avg_collisions = sum(coll_deque) / len(coll_deque)
    render_text("{:.1f} collisions".format(avg_collisions), 35)
    avg_checks = sum(checks_deque) / len(checks_deque)
    #render_text("{:.1f} checks".format(avg_checks), 60)
    entropy = np.linalg.norm(get_avg_pos_for_flag(physics_buckets.flat, "right")
        - get_avg_pos_for_flag(physics_buckets.flat, "left"))
    render_text("separation {:.2f}".format(entropy), 60)

    # if ticks % 10 == 0:
    #     with open("data.csv", mode="a") as csv_file:
    #         csv_writer = csv.writer(csv_file, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL)
    #         csv_writer.writerow([ticks, entropy])
    

    pygame.display.flip() # Display frame


