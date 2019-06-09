from collections import deque

import numpy as np
import pygame


# Freestanding utils
def calc_normal(arr):
    """Returns the input vector scaled to have a euclidic lenght of 1"""
    return np.array(arr) / np.linalg.norm(arr)

def clamp_index(n, lenght):
    """Return a valid index for a list of length length"""
    return max(min(n, lenght - 1), 0)

def is_valid_index(index, container):
    """Returns whether the index is valid for container"""
    i_arr = np.array(index)
    shape = np.array(container.shape)
    return (i_arr < shape).all() and (i_arr >= 0).all()

def indices(x, y):
    """Returns a list of 2d tuples adressing every index"""
    return [(i, j) for i in range(x) for j in range(y)]

def neighbor_buckets(index):
    """Returns a list of 2d tuples"""
    i, j = index
    return [(i, j), (i, j + 1), (i + 1, j), (i + 1, j + 1)]


# Classes
class Particle:
    """A class to represent a single ball-like particle"""
    def __init__(self, pos, vel, mass, radius, flag=""):
        self.pos = np.array(pos).astype(float)
        self.vel = np.array(vel).astype(float)
        self.mass = float(mass)
        self.radius = float(radius)
        self.flag = flag

    def get_energy(self):
        return .5 * self.mass * sum(self.vel ** 2) - self.mass * PHYSICS.gravity[1] * self.pos[1]

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
        # Quadratic interpolate the color based on the speed
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


# Graphic utils
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


# Init utils
def random_helium(energy):
    helium_mass = 4
    helium_radius = 6 # 31 pm
    helium_color_1 = (150, 150, 255)
    helium_color_2 = (255, 50, 50)
    if PHYSICS.gravity[1] != 0:
        # calculate maximal height so e_pos isn't > energy
        max_height = min(energy / abs(PHYSICS.gravity[1]) / helium_mass, screen_size[1])
    else:
        max_height = screen_size[1]
    pos = np.random.random(2) * np.array((screen_size[0], max_height))
    e_pot = abs(PHYSICS.gravity[1]) * helium_mass * pos[1]
    e_kin = energy - e_pot
    speed = np.sqrt(2 * e_kin / helium_mass)
    angle = np.random.rand() * np.pi * 2
    vel = np.array([np.cos(angle), np.sin(angle)]) * speed
    # Used to show diffusion
    color = helium_color_1 if pos[0] < screen_size[0] / 2 else helium_color_2
    flag = "left" if pos[0] < screen_size[0] / 2 else "right"
    return Colored_Particle(pos, vel, helium_mass, helium_radius, color, flag=flag)


def random_argon(energy):
    argon_mass = 40
    argon_radius = 14 # 71 pm
    argon_color = (242, 96, 213)
    if PHYSICS.gravity[1] != 0:
        max_height = min(energy / abs(PHYSICS.gravity[1]) / argon_mass, screen_size[1])
    else:
        max_height = screen_size[1]
    pos = np.random.random(2) * np.array((screen_size[0], max_height))
    e_pot = abs(PHYSICS.gravity[1]) * argon_mass * pos[1]
    e_kin = energy - e_pot
    speed = np.sqrt(2 * e_kin / argon_mass)
    angle = np.random.rand() * np.pi * 2
    vel = np.array([np.cos(angle), np.sin(angle)]) * speed
    return Colored_Particle(pos, vel, argon_mass, argon_radius, argon_color)


def calc_bucket(obj, buckets_shape, screen_size):
    buckets_x, buckets_y = buckets_shape
    bucket_size_x = screen_size[0] / buckets_x
    bucket_size_y = screen_size[1] / buckets_y

    new_bucket_x = clamp_index(int(obj.pos[0] // bucket_size_x), buckets_x)
    new_bucket_y = clamp_index(int(obj.pos[1] // bucket_size_y), buckets_y)

    return (new_bucket_x, new_bucket_y)


# Static class to define Physiscs constants
class PHYSICS:
    gravity = np.array([0., -50.])

# Init Pygame
pygame.init()
pygame.font.init()

screen_size = width, height = 900, 700

screen = pygame.display.set_mode(screen_size, pygame.RESIZABLE)
pygame.display.set_caption("Simulation")
clock = pygame.time.Clock()

# Init Objects
bucket_count = buckets_x, buckets_y = 15, 12
physics_buckets = np.empty(bucket_count, dtype=list)
for i, j in indices(buckets_x, buckets_y):
    physics_buckets[i, j] = list()

# Generate helium objects
for obj in [random_helium(e) for e in np.ones(100) * 200_000]:
    b = calc_bucket(obj, bucket_count, screen_size)
    physics_buckets[b].append(obj)

# Generate helium objects
for obj in [random_argon(e) for e in np.ones(50) * 5_000_000]:
    b = calc_bucket(obj, bucket_count, screen_size)
    physics_buckets[b].append(obj)

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

    # Pause functionality
    if not simulating:
        continue

    ticks += 1

    # Move every Particle and apply gravity
    for bucket in physics_buckets.flat:
        for obj in bucket:
            obj.tick(dt)

    # Update the buckets objects lie in
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
    # For every bucket...
    for bucket_index in indices(buckets_x, buckets_y):
        # For the 4 buckets around the lower right corner..
        for other_index in neighbor_buckets(bucket_index):
            if not is_valid_index(other_index, physics_buckets):
                continue
            # For all objets in the current bucket..
            for i, obj1 in enumerate(physics_buckets[bucket_index]):
                # Prevents double calculation where obj1 and obj2 are swapped
                start_index = i + 1 if bucket_index == other_index else 0
                # For all objects in the other bucket or the rest in the current
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
                    vel_2 = obj2.vel + mass_scalar_2 * dot_scalar * pos_diff
                    obj1.vel = vel_1
                    obj2.vel = vel_2


    # Draw background
    screen.fill((255, 255, 255))

    # Draw histogram
    speed_hist = get_speed_histogram(physics_buckets.flat)
    if total_speed_hist is None:
        total_speed_hist = speed_hist
    else:
        total_speed_hist += speed_hist
    draw_histogram(screen, total_speed_hist)

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

    # Calculate some output values
    simulation_speed = 1 / frame_time
    avg_collisions = sum(coll_deque) / len(coll_deque)
    avg_checks = sum(checks_deque) / len(checks_deque)
    separation = np.linalg.norm(get_avg_pos_for_flag(physics_buckets.flat, "right")
        - get_avg_pos_for_flag(physics_buckets.flat, "left"))
    total_energy = sum([obj.get_energy for bucket in physics_buckets.flat for obj in bucket])

    # Display info values
    render_text("tick {} @ {:.3f} t/s".format(ticks, simulation_speed), 10)
    render_text("{:.1f} collisions/t".format(avg_collisions), 35)
    render_text("{:.1f} checks/t".format(avg_checks), 60)
    render_text("separation {:.2f}".format(separation), 85)
    render_text("total energy {:.2E}".format(separation), 110)

    # Display frame
    pygame.display.flip()


